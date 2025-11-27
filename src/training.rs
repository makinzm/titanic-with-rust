use anyhow::Result;
use lgbm::{
    parameters::{Objective, Verbosity},
    Booster, Dataset, Field, MatBuf, Parameters,
};
use polars::series::Series;
use polars::prelude::*;
use std::sync::Arc;
use rand::seq::SliceRandom;
use rand::rngs::SmallRng;
use rand::SeedableRng;

// コンパイル時に特徴量数を確定し、パフォーマンスを向上
const FEATURE_COUNT: usize = 7;
const FEATURE_NAMES: [&str; FEATURE_COUNT] = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone"];

pub fn convert_df_to_vec(df: &DataFrame) -> Result<Vec<[f64; FEATURE_COUNT]>> {
    // 事前に列を取得してキャッシュ（パフォーマンス向上）
    let columns: Result<Vec<_>> = FEATURE_NAMES
        .iter()
        .map(|&name| {
            df.column(name)
                .map_err(|e| anyhow::anyhow!("Column '{}' not found: {}", name, e))
        })
        .collect();
    let columns = columns?;
    
    // メモリを事前に確保（パフォーマンス向上）
    let mut result = Vec::with_capacity(df.height());
    
    for idx in 0..df.height() {
        let mut row = [0.0f64; FEATURE_COUNT];
        for (col_idx, column) in columns.iter().enumerate() {
            let value = column
                .get(idx)
                .map_err(|e| anyhow::anyhow!("Failed to get value at index {}: {}", idx, e))?
                .try_extract::<f64>()
                .map_err(|e| anyhow::anyhow!("Failed to convert value to f64 at column '{}', row {}: {}", FEATURE_NAMES[col_idx], idx, e))?;
            row[col_idx] = value;
        }
        result.push(row);
    }
    
    Ok(result)
}

#[macro_export]
macro_rules! train_model {
    ($train_data_path:expr, $output_model_path:expr, $oof_predictions_path:expr, $seed:expr) => {
        train_model($train_data_path, $output_model_path, $oof_predictions_path, $seed)
    };
    ($train_data_path:expr, $output_model_path:expr, $oof_predictions_path:expr) => {
        train_model($train_data_path, $output_model_path, $oof_predictions_path, None)
    };
}

pub fn train_model(
    train_data_path: &str,
    output_model_path: &str,
    oof_predictions_path: &str,
    seed: Option<u64>,
) -> Result<()> {
    // LazyFrameで直接読み込み、一度にすべての列をFloat64にキャスト（パフォーマンス向上）
    let df = LazyFrame::scan_csv(
        std::path::Path::new(train_data_path),
        ScanArgsCSV {
            has_header: true,
            ..Default::default()
        },
    )?
    .with_columns([
        col("Pclass").cast(DataType::Float64),
        col("Sex").cast(DataType::Float64),
        col("Age").cast(DataType::Float64),
        col("Fare").cast(DataType::Float64),
        col("Embarked").cast(DataType::Float64),
        col("FamilySize").cast(DataType::Float64),
        col("IsAlone").cast(DataType::Float64),
        col("Survived").cast(DataType::Float64),
    ])
    .collect()?;

    #[cfg(debug_assertions)]
    println!("Loaded data: {:?}", df.head(Some(5)));

    // 特徴量とラベルを分割
    let label_col = "Survived";
    let feature_data = convert_df_to_vec(&df)?;
    let data_matrix = MatBuf::from_rows(feature_data);

    let labels: Result<Vec<f32>> = df
        .column(label_col)?
        .f64()?
        .into_iter()
        .enumerate()
        .map(|(idx, v)| {
            v.ok_or_else(|| anyhow::anyhow!("Null value found in label column at index {}", idx))
                .map(|v| v as f32)
        })
        .collect();
    let labels = labels?;

    // 5分割交差検証
    let k = 5;
    let mut indices: Vec<usize> = (0..labels.len()).collect();

    let seed_value = seed.unwrap_or(42);
    indices.shuffle(&mut SmallRng::seed_from_u64(seed_value));
    let fold_size = labels.len() / k;

    let mut oof_predictions = vec![0.0; labels.len()];
    let mut accuracies = vec![];
    for fold in 0..k {
        println!("Training fold {}/{}", fold + 1, k);

        // トレーニングデータと検証データに分割
        let start = fold * fold_size + usize::min(fold, labels.len() % k);
        let end = (fold + 1) * fold_size + usize::min(fold + 1, labels.len() % k);

        let valid_idx = indices[start..end].iter().copied().collect::<Vec<_>>();
        let train_idx = indices[..start]
            .iter()
            .chain(indices[end..].iter())
            .copied()
            .collect::<Vec<_>>();

        // メモリを事前に確保してパフォーマンス向上
        let mut train_features = Vec::with_capacity(train_idx.len());
        let mut train_labels = Vec::with_capacity(train_idx.len());
        
        for &idx in &train_idx {
            train_features.push(
                data_matrix
                    .row(idx)
                    .try_into()
                    .expect("Row length mismatch")
            );
            train_labels.push(labels[idx]);
        }

        let mut valid_features = Vec::with_capacity(valid_idx.len());
        let mut valid_labels = Vec::with_capacity(valid_idx.len());
        
        for &idx in &valid_idx {
            valid_features.push(
                data_matrix
                    .row(idx)
                    .try_into()
                    .expect("Row length mismatch")
            );
            valid_labels.push(labels[idx]);
        }

        // LightGBM データセットの作成
        let mut train_dataset = Dataset::from_mat(
            &MatBuf::from_rows(train_features),
            None,
            &Parameters::new(),
        )?;
        train_dataset.set_field(Field::LABEL, &train_labels)?;

        // let mut valid_dataset = Dataset::from_mat(
        //     &MatBuf::from_rows(valid_features.clone()),
        //     None,
        //     &Parameters::new(),
        // )?;
        // valid_dataset.set_field(Field::LABEL, &valid_labels)?;

        // LightGBM パラメータ設定
        let mut params = Parameters::new();
        params.push("objective", Objective::Binary);
        params.push("verbosity", Verbosity::Fatal);
        params.push("num_iterations", 100);

        // モデルの作成とトレーニング
        let mut booster = Booster::new(Arc::new(train_dataset), &params)?;
        // booster.add_valid_data(Arc::new(valid_dataset))?;
        for _ in 0..100 {
            if booster.update_one_iter()? {
                break;
            }
        }

        // 検証データで予測
        let valid_predictions = booster.predict_for_mat(
            &MatBuf::from_rows(valid_features.clone()),
            lgbm::PredictType::Normal,
            0,
            None,
            &Parameters::default(),
        )?;

        for (i, &pred) in valid_idx.iter().zip(valid_predictions.values().iter()) {
            oof_predictions[*i] = pred;
        }

        // 精度計算を最適化（イテレータを使用）
        let correct = valid_predictions
            .values()
            .iter()
            .zip(valid_labels.iter())
            .filter(|(&pred, &label)| (pred > 0.5) == (label > 0.5))
            .count();
        let accuracy = correct as f32 / valid_labels.len() as f32;
        println!("Fold {}/{} - Accuracy: {:.4} ({:.2}%)", 
                 fold + 1, k, accuracy, accuracy * 100.0);

        accuracies.push(accuracy);
        
        // モデル保存のエラーハンドリング改善
        let fold_model_path = output_model_path.replace(".bin", &format!("_fold{}.bin", fold));
        booster.save_model(
            fold,
            None,
            lgbm::FeatureImportanceType::Split,
            std::path::Path::new(&fold_model_path),
        )
        .map_err(|e| anyhow::anyhow!("Failed to save model for fold {}: {}", fold, e))?;
    }

    // 全体精度計算と結果表示を最適化
    if accuracies.is_empty() {
        return Err(anyhow::anyhow!("No accuracies calculated - training failed"));
    }
    
    let overall_accuracy = accuracies.iter().sum::<f32>() / accuracies.len() as f32;
    let min_accuracy = accuracies.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_accuracy = accuracies.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    println!("\n=== Training Results ===");
    println!("Overall CV Accuracy: {:.4} ({:.2}%)", overall_accuracy, overall_accuracy * 100.0);
    println!("Min Fold Accuracy: {:.4} ({:.2}%)", min_accuracy, min_accuracy * 100.0);
    println!("Max Fold Accuracy: {:.4} ({:.2}%)", max_accuracy, max_accuracy * 100.0);
    println!("Std Dev: {:.4}", {
        let mean = overall_accuracy;
        let variance = accuracies.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / accuracies.len() as f32;
        variance.sqrt()
    });

    // OOF予測の保存を最適化
    let oof_series = Series::new("OOF_Predictions".into(), oof_predictions);
    let mut oof_df = DataFrame::new(vec![polars::prelude::Column::Series(oof_series)])
        .map_err(|e| anyhow::anyhow!("Failed to create OOF DataFrame: {}", e))?;

    let oof_file = std::fs::File::create(oof_predictions_path)
        .map_err(|e| anyhow::anyhow!("Failed to create OOF predictions file {}: {}", oof_predictions_path, e))?;
        
    CsvWriter::new(oof_file)
        .finish(&mut oof_df)
        .map_err(|e| anyhow::anyhow!("Failed to write OOF predictions: {}", e))?;

    println!("\nTraining completed successfully!");
    println!("Models saved with pattern: {}", output_model_path.replace(".bin", "_fold*.bin"));
    println!("OOF predictions saved to: {}", oof_predictions_path);

    Ok()
}


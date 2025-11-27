use anyhow::{Result, Context};
use lgbm::{
    parameters::{Objective, Verbosity},
    Booster, Dataset, Field, MatBuf, Parameters,
};
use polars::prelude::*;
use std::sync::Arc;
use rand::seq::SliceRandom;
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// 簡素化されたDataFrameからベクターへの変換
pub fn convert_df_to_vec(df: &DataFrame) -> Result<Vec<[f64; 7]>> {
    let features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone"];
    
    let mut result = Vec::new();
    for row in 0..df.height() {
        let mut feature_row = [0.0; 7];
        for (i, &col) in features.iter().enumerate() {
            let val = df.column(col)?.get(row)?;
            feature_row[i] = match val {
                AnyValue::Int64(v) => v as f64,
                AnyValue::Int32(v) => v as f64,
                AnyValue::Int16(v) => v as f64,
                AnyValue::Int8(v) => v as f64,
                AnyValue::UInt64(v) => v as f64,
                AnyValue::UInt32(v) => v as f64,
                AnyValue::UInt16(v) => v as f64,
                AnyValue::UInt8(v) => v as f64,
                AnyValue::Float64(v) => v,
                AnyValue::Float32(v) => v as f64,
                AnyValue::Boolean(v) => if v { 1.0 } else { 0.0 },
                _ => 0.0,
            };
        }
        result.push(feature_row);
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
    // Lazy読み込みでメモリ効率を最大化
    let df = LazyCsvReader::new(train_data_path)
        .with_has_header(true)
        .finish()?
        // データ型の最適化と一括変換
        .with_columns([
            // Booleanカラムの処理
            col("IsAlone").cast(DataType::Float64),
            // 整数カラムの処理
            col("Pclass").cast(DataType::Float64),
            col("Sex").cast(DataType::Float64),
            col("Embarked").cast(DataType::Float64),
            col("FamilySize").cast(DataType::Float64),
            // 浮動小数点カラムの処理
            col("Age").cast(DataType::Float64),
            col("Fare").cast(DataType::Float64),
            col("Survived").cast(DataType::Float64),
        ])
        .collect()
        .with_context(|| format!("Failed to load training data from {}", train_data_path))?;

    #[cfg(debug_assertions)]
    {
        println!("Loaded training data: {:?}", df.head(Some(5)));
        println!("Data shape: {:?}", df.shape());
        println!("Memory usage: {} bytes", df.estimated_size());
    }

    // 特徴量とラベルを効率的に分割
    let label_col = "Survived";
    let feature_data = convert_df_to_vec(&df)
        .with_context(|| "Failed to convert DataFrame to feature matrix")?;
    let data_matrix = MatBuf::from_rows(feature_data);

    // ラベルデータの効率的な抽出
    let labels: Result<Vec<f32>> = df
        .column(label_col)
        .with_context(|| format!("Label column '{}' not found", label_col))?
        .f64()
        .with_context(|| "Failed to convert labels to f64")?
        .into_iter()
        .map(|v| {
            v.ok_or_else(|| anyhow::anyhow!("Found null value in labels"))
                .map(|val| val as f32)
        })
        .collect();
    
    let labels = labels?;

    // K-fold交差検証の設定
    const K_FOLDS: usize = 5;
    let mut indices: Vec<usize> = (0..labels.len()).collect();

    let seed_value = seed.unwrap_or(42);
    indices.shuffle(&mut SmallRng::seed_from_u64(seed_value));
    
    let mut oof_predictions = vec![0.0; labels.len()];
    let mut fold_accuracies = Vec::with_capacity(K_FOLDS);
    
    for fold in 0..K_FOLDS {
        println!("Training fold {}/{}", fold + 1, K_FOLDS);

        // 効率的なフォールド分割
        let fold_size = labels.len() / K_FOLDS;
        let remainder = labels.len() % K_FOLDS;
        let start = fold * fold_size + fold.min(remainder);
        let end = (fold + 1) * fold_size + (fold + 1).min(remainder);

        let valid_idx: Vec<usize> = indices[start..end].to_vec();
        let train_idx: Vec<usize> = indices[..start]
            .iter()
            .chain(indices[end..].iter())
            .copied()
            .collect();

        // メモリ効率を考慮したデータ抽出
        let train_features: Result<Vec<[f64; 7]>> = train_idx
            .iter()
            .map(|&idx| {
                data_matrix
                    .row(idx)
                    .try_into()
                    .with_context(|| format!("Row length mismatch at index {}", idx))
            })
            .collect();
        let train_features = train_features?;

        let train_labels: Vec<f32> = train_idx
            .iter()
            .map(|&idx| labels[idx])
            .collect();

        let valid_features: Result<Vec<[f64; 7]>> = valid_idx
            .iter()
            .map(|&idx| {
                data_matrix
                    .row(idx)
                    .try_into()
                    .with_context(|| format!("Row length mismatch at validation index {}", idx))
            })
            .collect();
        let valid_features = valid_features?;

        let valid_labels: Vec<f32> = valid_idx
            .iter()
            .map(|&idx| labels[idx])
            .collect();

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

        // Calculate accuracy
        let correct = valid_predictions
            .values()
            .iter()
            .zip(valid_labels.iter())
            .filter(|(&pred, &label)| {
                if pred > 0.5 {
                    label > 0.5
                } else {
                    label < 0.5
                }
            })
            .count();
        let accuracy = correct as f32 / valid_labels.len() as f32;
        println!("Fold {} accuracy: {:.2}%", fold, accuracy * 100.0);

        fold_accuracies.push(accuracy);
        // Save model
        let output_model_path = output_model_path.replace(".bin", &format!("_fold{}.bin", fold));
        booster.save_model(
            fold,
            None,
            lgbm::FeatureImportanceType::Split,
            std::path::Path::new(&format!("{}", output_model_path)),
        )?;
    }

    // 結果の統計とレポート
    let mean_accuracy = fold_accuracies.iter().sum::<f32>() / fold_accuracies.len() as f32;
    let std_accuracy = {
        let variance = fold_accuracies
            .iter()
            .map(|&acc| (acc - mean_accuracy).powi(2))
            .sum::<f32>() / fold_accuracies.len() as f32;
        variance.sqrt()
    };
    
    println!("Cross-validation results:");
    println!("  Mean accuracy: {:.4} ± {:.4} ({:.2}% ± {:.2}%)", 
             mean_accuracy, std_accuracy, mean_accuracy * 100.0, std_accuracy * 100.0);
    println!("  Individual fold accuracies: {:?}", 
             fold_accuracies.iter().map(|&acc| format!("{:.4}", acc)).collect::<Vec<_>>());

    // OOF予測の効率的な保存
    if let Some(parent) = std::path::Path::new(oof_predictions_path).parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory for OOF predictions: {}", oof_predictions_path))?;
    }
    
    let oof_df = DataFrame::new(vec![
        Column::new("OOF_Predictions".into(), oof_predictions)
    ]).with_context(|| "Failed to create OOF predictions DataFrame")?;
    
    let mut oof_df = oof_df; // Make mutable for CsvWriter
    CsvWriter::new(std::fs::File::create(oof_predictions_path)
        .with_context(|| format!("Failed to create OOF predictions file: {}", oof_predictions_path))?)
        .include_header(true)
        .finish(&mut oof_df)
        .with_context(|| "Failed to write OOF predictions to CSV")?;

    println!("Training completed successfully!");
    println!("  Models saved with prefix: {}", output_model_path.replace(".bin", "_fold*.bin"));
    println!("  OOF predictions saved to: {}", oof_predictions_path);

    Ok(())
}


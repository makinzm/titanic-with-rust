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

pub fn convert_df_to_vec(
    df: &DataFrame,
) -> Vec<[f64; 7]> {

    let features: Vec<&str> = vec!["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone"];

    (0..df.height())
        .map(|idx| {
            features
                .iter()
                .map(|&col| {
                    df.column(col)
                        .unwrap()
                        .get(idx)
                        .unwrap()
                        .try_extract::<f64>()
                        .unwrap_or_else(|_| {
                            // もし f64 に変換できない場合、i64 として取得して f64 に変換
                            df.column(col)
                                .unwrap()
                                .get(idx)
                                .unwrap()
                                .try_extract::<i64>()
                                .unwrap_or(0) as f64
                        })
                })
                .collect::<Vec<f64>>()
                .try_into()
                .expect("Row length mismatch")
        })
        .collect()
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
    // データの読み込み
    let mut df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(train_data_path.into()))?
        .finish()?;

    // キャストする列のリスト
    let columns_to_cast = vec!["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone", "Survived"];

    // 各列を Float64 にキャスト
    for col_name in &columns_to_cast {
        df = df
            .lazy()
            .with_column(
                col(*col_name)
                .cast(DataType::Float64)
            )
            .collect()?;
    }

    println!("Loaded data: {:?}", df.head(Some(5)));

    // 特徴量とラベルを分割
    let label_col = "Survived";
    let feature_data = convert_df_to_vec(&df);
    let data_matrix = MatBuf::from_rows(feature_data);

    let labels = df
        .column(label_col)?
        .f64()?
        .into_iter()
        .map(|v| v.unwrap() as f32)
        .collect::<Vec<_>>();

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

        let train_features: Vec<[f64; 7]> = train_idx
            .iter()
            .map(|&idx| {
                data_matrix
                    .row(idx)
                    .try_into()
                    .expect("Row length mismatch")
            })
            .collect();


        let train_labels: Vec<f32> = train_idx
            .iter()
            .map(|&idx| labels[idx])
            .collect::<Vec<_>>();


        let valid_features: Vec<[f64; 7]> = valid_idx
            .iter()
            .map(|&idx| {
                data_matrix
                    .row(idx)
                    .try_into()
                    .expect("Row length mismatch")
            })
            .collect();

        let valid_labels: Vec<f32> = valid_idx
            .iter()
            .map(|&idx| labels[idx])
            .collect::<Vec<_>>();

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

        accuracies.push(accuracy);
        // Save model
        let output_model_path = output_model_path.replace(".bin", &format!("_fold{}.bin", fold));
        booster.save_model(
            fold,
            None,
            lgbm::FeatureImportanceType::Split,
            std::path::Path::new(&format!("{}", output_model_path)),
        )?;
    }

    // Calculate overall accuracy
    let overall_accuracy = accuracies.iter().sum::<f32>() / accuracies.len() as f32;
    println!("Overall accuracy: {:.2}%", overall_accuracy * 100.0);

    // OOF予測の保存
    let oof_series = Series::new("OOF_Predictions".into(), oof_predictions);
    let mut oof_df = DataFrame::new(vec![polars::prelude::Column::Series(oof_series)])?;

    CsvWriter::new(std::fs::File::create(oof_predictions_path)?)
        .finish(&mut oof_df)?;

    println!("Training complete. Saving model to {}", output_model_path);

    Ok(())
}


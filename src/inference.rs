use anyhow::Result;
use crate::feature_engineering::process_features;
use crate::training::{convert_df_to_vec, FEATURE_COUNT};
use lgbm::{
    Booster, MatBuf, Parameters,
};
use polars::prelude::*;
use std::path::{Path, PathBuf};

const NUM_FOLDS: usize = 5;

pub fn run_inference(model_path: &str, input_data_path: &str, output_data_path: &str) -> Result<()> {
    println!(
        "Running inference:\n  Model: {}\n  Input: {}\n  Output: {}",
        model_path, input_data_path, output_data_path
    );

    // ファイルパスをPathBufで最適化
    let model_paths: Result<Vec<PathBuf>> = (0..NUM_FOLDS)
        .map(|fold| {
            let path_str = model_path.replace(".bin", &format!("_fold{}.bin", fold));
            let path = PathBuf::from(&path_str);
            if !path.exists() {
                Err(anyhow::anyhow!("Model file does not exist: {}", path_str))
            } else {
                Ok(path)
            }
        })
        .collect();
    let model_paths = model_paths?;

    // 一時的な特徴量ファイルを作成（メモリ効率化）
    let output_feature_path = PathBuf::from(output_data_path)
        .with_file_name(
            PathBuf::from(output_data_path)
                .file_stem()
                .ok_or_else(|| anyhow::anyhow!("Invalid output file path: {}", output_data_path))?
                .to_string_lossy()
                + "_features.csv"
        );
    
    // 特徴量処理を実行
    process_features(
        input_data_path, 
        output_feature_path.to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid feature path encoding"))?,
    )
    .map_err(|e| anyhow::anyhow!("Feature processing failed: {}", e))?;

    // データ読み込みを最適化
    let df = LazyFrame::scan_csv(
        &output_feature_path,
        ScanArgsCSV {
            has_header: true,
            ..Default::default()
        },
    )?
    .collect()?;

    // PassengerIdを安全に取得
    let passenger_ids: Result<Vec<Option<i64>>> = df.column("PassengerId")
        .map_err(|e| anyhow::anyhow!("PassengerId column not found: {}", e))?
        .i64()
        .map_err(|e| anyhow::anyhow!("Failed to convert PassengerId to i64: {}", e))?
        .to_vec();
    let passenger_ids = passenger_ids?;

    // テストデータの特徴量を取得
    let test_data = convert_df_to_vec(&df)
        .map_err(|e| anyhow::anyhow!("Failed to convert DataFrame to features: {}", e))?;
    let test_data = MatBuf::from_rows(test_data);

    // 予測の実行（メモリ効率化）
    let num_samples = test_data.rows();
    let mut final_predictions = vec![0.0f64; num_samples];
    
    println!("Loading {} models and making predictions...", NUM_FOLDS);
    
    for (fold, model_path) in model_paths.iter().enumerate() {
        let (model, num_iteration) = Booster::from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load model {}: {}", model_path.display(), e))?;
            
        println!("  Fold {}/{}: {} iterations", fold + 1, NUM_FOLDS, num_iteration);
        
        let prediction = model.predict_for_mat(
            &test_data,
            lgbm::PredictType::Normal,
            0,
            None,
            &Parameters::default(),
        )
        .map_err(|e| anyhow::anyhow!("Prediction failed for fold {}: {}", fold, e))?;
        
        // 直接平均を計算（中間配列を作成しない）
        for (i, &value) in prediction.values().iter().enumerate() {
            final_predictions[i] += value / NUM_FOLDS as f64;
        }
    }

    // 予測結果（確率）を保存
    let predictions_path = PathBuf::from(output_data_path)
        .with_file_name(
            PathBuf::from(output_data_path)
                .file_stem()
                .unwrap()
                .to_string_lossy()
                + "_predictions.csv"
        );
        
    let mut predictions_df = DataFrame::new(vec![
        Column::new(
            "Survived_Probability".into(),
            Series::new("Survived_Probability".into(), final_predictions.clone()),
        ),
    ])
    .map_err(|e| anyhow::anyhow!("Failed to create predictions DataFrame: {}", e))?;
    
    let predictions_file = std::fs::File::create(&predictions_path)
        .map_err(|e| anyhow::anyhow!("Failed to create predictions file: {}", e))?;
    CsvWriter::new(predictions_file)
        .finish(&mut predictions_df)
        .map_err(|e| anyhow::anyhow!("Failed to write predictions: {}", e))?;

    // PassengerIdを安全に処理
    let valid_passenger_ids: Vec<i64> = passenger_ids
        .iter()
        .filter_map(|&id| id)
        .collect();
    
    // 最終結果（PassengerIdと二値化した予測）を保存
    let binary_predictions: Vec<i32> = final_predictions
        .iter()
        .take(valid_passenger_ids.len())  // PassengerIdと同じ長さまで
        .map(|&prob| if prob > 0.5 { 1 } else { 0 })
        .collect();
    
    if valid_passenger_ids.len() != binary_predictions.len() {
        return Err(anyhow::anyhow!(
            "Mismatch between valid PassengerIds ({}) and predictions ({})",
            valid_passenger_ids.len(),
            binary_predictions.len()
        ));
    }

    let mut final_df = DataFrame::new(vec![
        Column::new(
            "PassengerId".into(),
            Series::new("PassengerId".into(), valid_passenger_ids),
        ),
        Column::new(
            "Survived".into(),
            Series::new("Survived".into(), binary_predictions),
        ),
    ])
    .map_err(|e| anyhow::anyhow!("Failed to create final DataFrame: {}", e))?;

    let output_file = std::fs::File::create(output_data_path)
        .map_err(|e| anyhow::anyhow!("Failed to create output file: {}", e))?;
    CsvWriter::new(output_file)
        .finish(&mut final_df)
        .map_err(|e| anyhow::anyhow!("Failed to write final results: {}", e))?;

    // 一時ファイルをクリーンアップ
    if let Err(e) = std::fs::remove_file(&output_feature_path) {
        println!("Warning: Could not remove temporary file {:?}: {}", output_feature_path, e);
    }

    println!("\n=== Inference Results ===");
    println!("Processed {} samples", final_predictions.len());
    println!("Probability range: [{:.4}, {:.4}]", 
             final_predictions.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
             final_predictions.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
    println!("Predictions saved to: {}", output_data_path);
    println!("Probabilities saved to: {:?}", predictions_path);

    Ok(())
}


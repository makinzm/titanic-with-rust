use anyhow::{Result, Context};
use crate::feature_engineering::process_features;
use crate::training::convert_df_to_vec;
use lgbm::{
    Booster, MatBuf, Parameters,
};
use polars::prelude::*;
use std::path::Path;

pub fn run_inference(model_path: &str, input_data_path: &str, output_data_path: &str) -> Result<()> {
    println!(
        "Running inference: {} -> {}",
        input_data_path, output_data_path
    );

    // K-fold モデルパスの生成
    const K_FOLDS: usize = 5;
    let model_paths: Vec<String> = (0..K_FOLDS)
        .map(|fold| model_path.replace(".bin", &format!("_fold{}.bin", fold)))
        .collect();

    // モデルファイルの存在確認
    for (fold, path) in model_paths.iter().enumerate() {
        if !Path::new(path).exists() {
            return Err(anyhow::anyhow!(
                "Model file for fold {} not found: {}",
                fold, path
            ));
        }
    }

    // 特徴量エンジニアリングの実行
    let output_feature_path = output_data_path.replace(".csv", "_features.csv");
    process_features(input_data_path, &output_feature_path)
        .with_context(|| format!("Feature engineering failed for {}", input_data_path))?;

    // 効率的なデータ読み込み
    let df = LazyCsvReader::new(&output_feature_path)
        .with_has_header(true)
        .finish()?
        .select([
            col("PassengerId"),
            col("Pclass"),
            col("Sex"),
            col("Age"),
            col("Fare"),
            col("Embarked"),
            col("FamilySize"),
            col("IsAlone"),
        ])
        .collect()
        .with_context(|| format!("Failed to load processed features from {}", output_feature_path))?;

    // PassengerIdを抽出（エラーハンドリング付き）
    let passenger_ids: Vec<Option<i64>> = df
        .column("PassengerId")
        .with_context(|| "PassengerId column not found")?        
        .i64()
        .with_context(|| "Failed to convert PassengerId to i64")?        
        .into_iter()
        .collect();
    
    // Null値チェック
    if passenger_ids.iter().any(|id| id.is_none()) {
        return Err(anyhow::anyhow!("Found null values in PassengerId column"));
    }
    
    let passenger_ids: Vec<i64> = passenger_ids.into_iter().map(|id| id.unwrap()).collect();

    #[cfg(debug_assertions)]
    {
        println!("Test data shape: {:?}", df.shape());
        println!("Memory usage: {} bytes", df.estimated_size());
    }

    // テストデータの特徴量変換
    let test_features = convert_df_to_vec(&df)
        .with_context(|| "Failed to convert test data to feature matrix")?;
    let test_matrix = MatBuf::from_rows(test_features);

    // アンサンブル予測の実行
    let mut ensemble_predictions = vec![0.0; passenger_ids.len()];
    
    for (fold, model_path) in model_paths.iter().enumerate() {
        // モデル読み込み
        let (model, num_iterations) = Booster::from_file(Path::new(model_path))
            .with_context(|| format!("Failed to load model from {}", model_path))?;
        
        #[cfg(debug_assertions)]
        println!("Loaded fold {} model with {} iterations from {}", 
                fold, num_iterations, model_path);
        
        // 予測実行
        let fold_predictions = model.predict_for_mat(
            &test_matrix,
            lgbm::PredictType::Normal,
            0,
            None,
            &Parameters::default(),
        ).with_context(|| format!("Prediction failed for fold {} model", fold))?;
        
        // アンサンブル予測に加算
        for (i, &pred) in fold_predictions.values().iter().enumerate() {
            ensemble_predictions[i] += pred;
        }
    }
    
    // 平均化
    let final_predictions: Vec<f64> = ensemble_predictions
        .iter()
        .map(|&sum_pred| sum_pred / K_FOLDS as f64)
        .collect();

    // 出力ディレクトリの作成
    if let Some(parent) = Path::new(output_data_path).parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output directory for {}", output_data_path))?;
    }
    
    // 生の予測値も保存（デバッグ用）
    let predictions_raw_path = output_data_path.replace(".csv", "_predictions.csv");
    {
        let mut predictions_df = DataFrame::new(vec![
            Column::new("PassengerId".into(), passenger_ids.clone()),
            Column::new("Prediction_Raw".into(), final_predictions.clone()),
        ]).with_context(|| "Failed to create raw predictions DataFrame")?;
        
        CsvWriter::new(std::fs::File::create(&predictions_raw_path)
            .with_context(|| format!("Failed to create raw predictions file: {}", predictions_raw_path))?)
            .include_header(true)
            .finish(&mut predictions_df)
            .with_context(|| "Failed to write raw predictions to CSV")?;
    }
    
    // 最終提出用の予測結果（0/1に変換）
    let binary_predictions: Vec<i32> = final_predictions
        .iter()
        .map(|&prob| if prob > 0.5 { 1 } else { 0 })
        .collect();
    
    let mut submission_df = DataFrame::new(vec![
        Column::new("PassengerId".into(), passenger_ids),
        Column::new("Survived".into(), binary_predictions.clone()),
    ]).with_context(|| "Failed to create submission DataFrame")?;
    
    CsvWriter::new(std::fs::File::create(output_data_path)
        .with_context(|| format!("Failed to create submission file: {}", output_data_path))?)
        .include_header(true)
        .finish(&mut submission_df)
        .with_context(|| "Failed to write submission to CSV")?;
    
    // 結果サマリー
    let survival_rate = binary_predictions.iter().map(|&x| x as f64).sum::<f64>() / binary_predictions.len() as f64;
    
    println!("Inference completed successfully!");
    // println!("  Processed {} passengers", passenger_ids.len());
    println!("  Predicted survival rate: {:.2}%", survival_rate * 100.0);
    println!("  Raw predictions saved to: {}", predictions_raw_path);
    println!("  Final submission saved to: {}", output_data_path);
    
    // 一時的な特徴量ファイルを削除
    if Path::new(&output_feature_path).exists() {
        std::fs::remove_file(&output_feature_path)
            .with_context(|| format!("Failed to remove temporary file: {}", output_feature_path))?;
    }
    
    Ok(())
}


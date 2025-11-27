mod feature_engineering;
mod training;
mod inference;

use anyhow::{Result, Context};
use training::train_model;
use std::time::Instant;

fn pipeline(
    train_csv: &str,
    train_features_csv: &str,
    oof_predictions_csv: &str,
    model_path: &str,
    test_csv: &str,
    submission_csv: &str,
) -> Result<()> {
    println!("=== Titanic Survival Prediction Pipeline ===");
    
    // Step 1: Feature Engineering
    {
        let start_time = Instant::now();
        println!("\n[1/3] Running feature engineering...");
        feature_engineering::process_features(train_csv, train_features_csv)
            .with_context(|| "Feature engineering step failed")?;
        println!("    ✓ Feature engineering completed in {:.2}s", start_time.elapsed().as_secs_f64());
    }
    
    // Step 2: Model Training
    {
        let start_time = Instant::now();
        println!("\n[2/3] Training models with cross-validation...");
        train_model!(train_features_csv, model_path, oof_predictions_csv)
            .with_context(|| "Model training step failed")?;
        println!("    ✓ Model training completed in {:.2}s", start_time.elapsed().as_secs_f64());
    }
    
    // Step 3: Inference
    {
        let start_time = Instant::now();
        println!("\n[3/3] Running inference on test data...");
        inference::run_inference(model_path, test_csv, submission_csv)
            .with_context(|| "Inference step failed")?;
        println!("    ✓ Inference completed in {:.2}s", start_time.elapsed().as_secs_f64());
    }
    
    Ok(())
}

fn main() -> Result<()> {
    let total_start_time = Instant::now();
    
    // 必要なディレクトリの作成
    std::fs::create_dir_all("intermediate")
        .with_context(|| "Failed to create intermediate directory")?;
    
    // ファイルパスの定義
    const TRAIN_CSV: &str = "datasets/train.csv";
    const TEST_CSV: &str = "datasets/test.csv";
    const TRAIN_FEATURES_CSV: &str = "intermediate/train_features.csv";
    const OOF_PREDICTIONS_CSV: &str = "intermediate/oof_predictions.csv";
    const MODEL_PATH: &str = "intermediate/model.bin";
    const SUBMISSION_CSV: &str = "intermediate/submission.csv";
    
    // 入力ファイルの存在確認
    for (name, path) in &[("Training data", TRAIN_CSV), ("Test data", TEST_CSV)] {
        if !std::path::Path::new(path).exists() {
            return Err(anyhow::anyhow!("Required file not found: {} ({})", name, path));
        }
    }
    
    // パイプライン実行
    match pipeline(
        TRAIN_CSV,
        TRAIN_FEATURES_CSV,
        OOF_PREDICTIONS_CSV,
        MODEL_PATH,
        TEST_CSV,
        SUBMISSION_CSV,
    ) {
        Ok(()) => {
            let total_time = total_start_time.elapsed();
            println!("\n=== Pipeline Completed Successfully! ===");
            println!("Total execution time: {:.2}s", total_time.as_secs_f64());
            println!("\nOutput files:");
            println!("  • Training features: {}", TRAIN_FEATURES_CSV);
            println!("  • OOF predictions: {}", OOF_PREDICTIONS_CSV);
            println!("  • Trained models: {}_fold*.bin", MODEL_PATH.replace(".bin", ""));
            println!("  • Final submission: {}", SUBMISSION_CSV);
            
            // ファイルサイズのレポート
            if let Ok(metadata) = std::fs::metadata(SUBMISSION_CSV) {
                println!("  • Submission file size: {} bytes", metadata.len());
            }
        }
        Err(e) => {
            eprintln!("\n❌ Pipeline failed after {:.2}s", total_start_time.elapsed().as_secs_f64());
            eprintln!("Error: {:?}", e);
            
            // エラーの原因チェーンを表示
            let mut source = e.source();
            let mut level = 1;
            while let Some(err) = source {
                eprintln!("  {}: {}", level, err);
                source = err.source();
                level += 1;
            }
            
            return Err(e);
        }
    }
    
    Ok(())
}


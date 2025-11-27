mod feature_engineering;
mod training;
mod inference;

use anyhow::Result;
use std::path::Path;
use std::time::Instant;
use training::train_model;

/// メインの機械学習パイプラインを実行します
fn pipeline(
    train_csv: &str,
    train_features_csv: &str,
    oof_predictions_csv: &str,
    model_path: &str,
    test_csv: &str,
    submission_csv: &str,
) -> Result<()> {
    // 入力ファイルの存在をチェック
    for (path, name) in [(train_csv, "Training data"), (test_csv, "Test data")] {
        if !Path::new(path).exists() {
            return Err(anyhow::anyhow!("{} file not found: {}", name, path));
        }
    }
    
    println!("=== Starting Titanic ML Pipeline ===");
    
    // ステップ 1: 特徴量エンジニアリング
    let start_time = Instant::now();
    println!("\n[1/3] Feature Engineering...");
    feature_engineering::process_features(train_csv, train_features_csv)
        .map_err(|e| anyhow::anyhow!("Feature engineering failed: {}", e))?;
    println!("    Completed in {:.2}s", start_time.elapsed().as_secs_f64());
    
    // ステップ 2: モデルトレーニング
    let start_time = Instant::now();
    println!("\n[2/3] Model Training...");
    train_model!(train_features_csv, model_path, oof_predictions_csv)
        .map_err(|e| anyhow::anyhow!("Model training failed: {}", e))?;
    println!("    Completed in {:.2}s", start_time.elapsed().as_secs_f64());
    
    // ステップ 3: 推論
    let start_time = Instant::now();
    println!("\n[3/3] Inference...");
    inference::run_inference(model_path, test_csv, submission_csv)
        .map_err(|e| anyhow::anyhow!("Inference failed: {}", e))?;
    println!("    Completed in {:.2}s", start_time.elapsed().as_secs_f64());
    
    Ok(())
}

fn main() -> Result<()> {
    let total_start = Instant::now();
    
    // フォルダ構成に基づいたファイルパス（constで最適化）
    const TRAIN_CSV: &str = "datasets/train.csv";
    const TEST_CSV: &str = "datasets/test.csv";
    const TRAIN_FEATURES_CSV: &str = "intermediate/train_features.csv";
    const OOF_PREDICTIONS_CSV: &str = "intermediate/oof_predictions.csv";
    const MODEL_PATH: &str = "intermediate/model.bin";
    const SUBMISSION_CSV: &str = "intermediate/submission.csv";

    // 中間ディレクトリを作成（必要に応じて）
    if let Some(parent) = Path::new(TRAIN_FEATURES_CSV).parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)
                .map_err(|e| anyhow::anyhow!("Failed to create intermediate directory: {}", e))?;
            println!("Created intermediate directory: {:?}", parent);
        }
    }

    // パイプライン実行
    pipeline(
        TRAIN_CSV,
        TRAIN_FEATURES_CSV,
        OOF_PREDICTIONS_CSV,
        MODEL_PATH,
        TEST_CSV,
        SUBMISSION_CSV,
    )
    .map_err(|e| {
        eprintln!("Pipeline failed: {}", e);
        e
    })?;

    let total_time = total_start.elapsed();
    println!("\n=== Pipeline Completed Successfully! ===");
    println!("Total execution time: {:.2}s", total_time.as_secs_f64());
    println!("\nOutput files:");
    println!("  - Submission: {}", SUBMISSION_CSV);
    println!("  - OOF Predictions: {}", OOF_PREDICTIONS_CSV);
    println!("  - Models: {}", MODEL_PATH.replace(".bin", "_fold*.bin"));
    
    Ok(())
}


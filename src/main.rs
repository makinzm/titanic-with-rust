mod feature_engineering;
mod training;
mod inference;

use anyhow::Result;
use training::train_model;

fn pipeline(
    train_csv: &str,
    train_features_csv: &str,
    oof_predictions_csv: &str,
    model_path: &str,
    test_csv: &str,
    submission_csv: &str,
) -> Result<()> {
    // 各ステップの実行
    feature_engineering::process_features(train_csv, train_features_csv)?;
    train_model!(train_features_csv, model_path, oof_predictions_csv)?;
    inference::run_inference(model_path, test_csv, submission_csv)?;
    Ok(())
}

fn main() -> Result<()> {
    println!("Starting Titanic challenge with Rust!");

    // フォルダ構成に基づいたファイルパス
    let train_csv = "datasets/train.csv";
    let train_features_csv = "intermediate/train_features.csv";

    let oof_predictions_csv = "intermediate/oof_predictions.csv";
    let model_path = "intermediate/model.bin";
    let test_csv = "datasets/test.csv";
    let submission_csv = "intermediate/submission.csv";

    pipeline(
        train_csv,
        train_features_csv,
        oof_predictions_csv,
        model_path,
        test_csv,
        submission_csv,
    )?;

    println!("Pipeline completed successfully!");
    
    Ok(())
}


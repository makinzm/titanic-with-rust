use anyhow::Result;

pub fn create_submission(predictions_path: &str, output_path: &str) -> Result<()> {
    println!(
        "Creating submission file using predictions from {}",
        predictions_path
    );
    // 提出用ファイルの作成
    Ok(())
}


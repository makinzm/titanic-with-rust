use anyhow::Result;

pub fn run_inference(model_path: &str, input_data_path: &str, output_data_path: &str) -> Result<()> {
    println!(
        "Running inference using model {} and input data {}",
        model_path, input_data_path
    );
    // 推論結果の保存処理
    Ok(())
}


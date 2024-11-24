use anyhow::Result;
use polars::prelude::*;

pub fn process_features(input_path: &str, output_path: &str) -> Result<()> {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(input_path.into()))?
        .finish()?;

    // 簡単な特徴量エンジニアリングの例
    let mut df = df
        .lazy()
        .with_column(col("Age").fill_nan(lit(30))) // 欠損値を埋める
        .collect()?;

    let file = std::fs::File::create(output_path).unwrap();
    CsvWriter::new(file)
        .finish(&mut df)
        .unwrap();

    Ok(())
}


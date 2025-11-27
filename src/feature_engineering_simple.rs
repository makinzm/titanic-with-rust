use anyhow::Result;
use polars::prelude::*;
use std::path::Path;

/// 簡素化された特徴量エンジニアリング関数
pub fn process_features(input_path: &str, output_path: &str) -> Result<()> {
    // 出力ディレクトリの作成
    if let Some(parent) = Path::new(output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    // CSVファイルの読み込み
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(input_path.into()))?
        .finish()?;

    // LazyFrameに変換して効率的な処理
    let df = df
        .lazy()
        .with_columns([
            // 性別の変換 (female -> 1, male -> 0)
            when(col("Sex").eq(lit("female")))
                .then(lit(1))
                .otherwise(lit(0))
                .alias("Sex"),
            
            // Embarkedの変換 (S -> 0, C -> 1, Q -> 2, null -> 0)
            col("Embarked")
                .fill_null(lit("S"))
                .replace(lit("S"), lit(0))
                .replace(lit("C"), lit(1)) 
                .replace(lit("Q"), lit(2))
                .alias("Embarked"),
            
            // Fareの欠損値を平均で埋める
            col("Fare").fill_null(col("Fare").mean()),
            
            // Ageの欠損値を中央値で埋める
            col("Age").fill_null(col("Age").median()),
            
            // 家族サイズの計算
            (col("Parch") + col("SibSp") + lit(1)).alias("FamilySize"),
        ])
        .with_columns([
            // IsAloneフラグ
            when(col("FamilySize").eq(lit(1)))
                .then(lit(true))
                .otherwise(lit(false))
                .alias("IsAlone"),
        ])
        .drop(["Name", "Ticket", "Cabin", "Parch", "SibSp"])
        .collect()?;

    // CSVファイルとして保存
    let mut df = df;
    CsvWriter::new(std::fs::File::create(output_path)?)
        .include_header(true)
        .finish(&mut df)?;

    println!("特徴量エンジニアリング完了: {}", output_path);
    Ok(())
}
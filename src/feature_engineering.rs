use anyhow::Result;
use polars::prelude::*;
use polars::lazy::dsl::*;
use polars::datatypes::DataType;

pub fn process_features(input_path: &str, output_path: &str) -> Result<()> {
    // データフレームをCSVから読み込む
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(input_path.into()))?
        .finish()?;

    // Check df
    println!("{:?}", df.head(Some(5)));

    // LazyFrameに変換
    let mut df = df.lazy();

    // 性別を数値に変換 (female -> 1, male -> 0)
    df = df.with_column(
        col("Sex")
            .replace(lit("female"), lit("1"))
            .replace(lit("male"), lit("0"))
            .cast(DataType::Int32),
    );

    // Embarkedを数値に変換 (null埋め, S -> 0, C -> 1, Q -> 2)
    df = df.with_column(
        col("Embarked")
            .fill_null(lit("S"))
            .replace(lit("S"), lit("0"))
            .replace(lit("C"), lit("1"))
            .replace(lit("Q"), lit("2"))
            .cast(DataType::Int32),
    );

    // Fareの欠損値を平均値で埋める
    df = df.with_column(
        col("Fare")
            .fill_null(col("Fare").mean())
    );

    // Ageの欠損値を中央値で埋める
    df = df.with_column(
        col("Age")
            .fill_null(col("Age").median()),
    );

    // 家族のサイズを計算 (Parch + SibSp + 1)
    df = df.with_column(
        (col("Parch") + col("SibSp") + lit(1)).alias("FamilySize"),
    );

    // IsAloneフラグを作成 (FamilySizeが1の場合は1、そうでない場合は0)
    df = df.with_column(
        when(col("FamilySize").eq(lit(1)))
            .then(lit(1))
            .otherwise(lit(0))
            .alias("IsAlone"),
    );

    // 不要な列を削除
    let columns_to_drop = vec!["Name", "Ticket", "Cabin", "Parch", "SibSp"];

    df = df.drop(columns_to_drop);

    // データフレームを収集し、CSVとして保存
    let mut df = df.collect()?;

    // Check df
    println!("Check df");
    println!("{:?}", df.head(Some(5)));

    let file = std::fs::File::create(output_path).unwrap();
    CsvWriter::new(file)
        .finish(&mut df)
        .unwrap();

    Ok(())
}


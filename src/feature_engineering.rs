use anyhow::Result;
use polars::prelude::*;
use std::path::Path;

// Ref: [[polars] python-kaggle-start-book-ch02_05](https://www.kaggle.com/code/sishihara/polars-python-kaggle-start-book-ch02-05)
pub fn process_features(input_path: &str, output_path: &str) -> Result<()> {
    // LazyFrameで直接読み込み（メモリ効率向上）
    let df = LazyFrame::scan_csv(
        Path::new(input_path),
        ScanArgsCSV {
            has_header: true,
            ..Default::default()
        },
    )?;

    // 複数の列変換を一度に実行（パフォーマンス向上）
    let df = df.with_columns([
        // 性別を数値に変換 (female -> 1, male -> 0) - より効率的なマッピング
        col("Sex")
            .map(
                |s| {
                    Ok(Some(
                        s.str()?
                            .into_iter()
                            .map(|opt_s| {
                                opt_s.map(|s| if s == "female" { 1i32 } else { 0i32 })
                            })
                            .collect::<Int32Chunked>()
                            .into_series(),
                    ))
                },
                GetOutput::from_type(DataType::Int32),
            )
            .alias("Sex"),
        
        // Embarkedを数値に変換 (null埋め, S -> 0, C -> 1, Q -> 2)
        col("Embarked")
            .fill_null(lit("S"))
            .map(
                |s| {
                    Ok(Some(
                        s.str()?
                            .into_iter()
                            .map(|opt_s| {
                                opt_s.map(|s| match s {
                                    "S" => 0i32,
                                    "C" => 1i32,
                                    "Q" => 2i32,
                                    _ => 0i32,
                                })
                            })
                            .collect::<Int32Chunked>()
                            .into_series(),
                    ))
                },
                GetOutput::from_type(DataType::Int32),
            )
            .alias("Embarked"),
        
        // Fareの欠損値を平均値で埋める
        col("Fare").fill_null(col("Fare").mean()),
        
        // Ageの欠損値を中央値で埋める  
        col("Age").fill_null(col("Age").median()),
        
        // 家族のサイズを計算 (Parch + SibSp + 1)
        (col("Parch") + col("SibSp") + lit(1)).alias("FamilySize"),
    ])
    // IsAloneフラグを作成 (FamilySizeが1の場合は1、そうでない場合は0)
    .with_column(
        when(col("FamilySize").eq(lit(1)))
            .then(lit(1i32))
            .otherwise(lit(0i32))
            .alias("IsAlone"),
    );

    // 不要な列を削除して、必要な列のみを選択（より効率的）
    let df = df.select([
        col("PassengerId"),
        col("Survived"),
        col("Pclass"),
        col("Sex"),
        col("Age"),
        col("Fare"),
        col("Embarked"),
        col("FamilySize"),
        col("IsAlone"),
    ]);

    // データフレームを収集し、CSVとして保存
    let mut df = df.collect()?;

    // デバッグ出力（必要に応じて）
    #[cfg(debug_assertions)]
    {
        println!("Processed features - first 5 rows:");
        println!("{:?}", df.head(Some(5)));
    }

    // エラーハンドリングを改善
    let file = std::fs::File::create(output_path)
        .map_err(|e| anyhow::anyhow!("Failed to create output file {}: {}", output_path, e))?;
        
    CsvWriter::new(file)
        .finish(&mut df)
        .map_err(|e| anyhow::anyhow!("Failed to write CSV: {}", e))?;

    Ok(())
}


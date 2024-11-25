use anyhow::Result;
use crate::feature_engineering::process_features;
use crate::training::convert_df_to_vec;
use lgbm::{
    Booster, MatBuf, Parameters,
};
use polars::prelude::*;
use std::path::Path;

pub fn run_inference(model_path: &str, input_data_path: &str, output_data_path: &str) -> Result<()> {
    println!(
        "Running inference using model {} and input data {}",
        model_path, input_data_path
    );

    // モデルの読み込み
    let model_pathes = (0..5)
        .map(|fold| model_path.replace(
            ".bin", 
            &format!("_fold{}.bin", fold)
        ))
        .collect::<Vec<_>>();

    let output_feature_path = output_data_path.replace(
        ".csv", 
        "_features.csv"
    );
    let _ = process_features(
        input_data_path, 
        &output_feature_path,
    );

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(
            Some(output_feature_path.into())
        )?
        .finish()?;

    let passenger_ids = df.column("PassengerId")
        .unwrap()
        .i64()?
        .to_vec();

    // テストデータの特徴量を取得
    let test_data = convert_df_to_vec(&df);
    let test_data = MatBuf::from_rows(
        test_data,
    );

    // 予測の実行
    // それぞれのfoldの平均を取る
    let mut predictions: Vec<Vec<f64>> = Vec::new();
    for (fold, model_path) in model_pathes.iter().enumerate() {
        let (model, num_iteration) = Booster::from_file(
            Path::new(model_path),
        )?;
        println!("モデル {} は {} イテレーションを持っています。", 
            fold, 
            num_iteration
        );
        let prediction = model.predict_for_mat(
            &test_data,
            lgbm::PredictType::Normal,
            0,
            None,
            &Parameters::default(),
        )?;
        predictions.push(prediction.values().to_vec());
    }

    // 予測の平均を取る
    let mut final_predictions = vec![0.0; predictions[0].len()];
    for prediction in &predictions {
        for (i, &value) in prediction.iter().enumerate() {
            final_predictions[i] += value;
        }
    }
    for value in &mut final_predictions {
        *value /= predictions.len() as f64;
    }

    // 予測結果をDataFrameに変換
    let mut df = DataFrame::new(vec![
        Column::new(
            "Survived".into(),
            Series::new(
                "Survived".into(),
                final_predictions.clone(),
            ),
        ),
    ])?;

    // 予測結果をCSVに保存
    CsvWriter::new(
        std::fs::File::create(
            output_data_path.replace(".csv", "_predictions.csv")
        )?,
    ).finish(&mut df)?;


    // PassengerIdと予測結果を結合
    // 予測結果を0, 1に変換
    let mut df = DataFrame::new(vec![
        Column::new(
            "PassengerId".into(),
            Series::new(
                "PassengerId".into(),
                passenger_ids,
            ),
        ),
        Column::new(
            "Survived".into(),
            Series::new(
                "Survived".into(),
                final_predictions.iter().map(
                    |&x| if x > 0.5 { 1 } else { 0 }
                ).collect::<Vec<_>>(),
            ),
        ),
    ])?;

    // 予測結果をCSVに保存
    CsvWriter::new(
        std::fs::File::create(output_data_path)?,
    ).finish(&mut df)?;

    Ok(())
}


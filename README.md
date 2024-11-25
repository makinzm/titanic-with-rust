# titanic-with-rust


```shell
export LIGHTGBM_LIB_DIR=$(pwd)/LightGBM
export LD_LIBRARY_PATH=$(pwd)/LightGBM:$LD_LIBRARY_PATH
```

```shell
cargo run main
```

Submit `intermediate/submission.csv` to [Titanic - Machine Learning from Disaster | Kaggle](https://www.kaggle.com/competitions/titanic/submissions)

(score: 0.72966)
---

# LightGBM Settings
[lgbm - crates.io: Rust Package Registry](https://crates.io/crates/lgbm)

```shell
git submodule update --init --recursive
```

1. Install cmake
  1. using devbox: [Introduction | Jetify Docs](https://www.jetify.com/docs/devbox/))
  1. [devbox add | Jetify Docs](https://www.jetify.com/docs/devbox/cli_reference/devbox_add/)
  1. [devbox global add | Jetify Docs](https://www.jetify.com/docs/devbox/cli_reference/devbox_global_add/)
  1. [cmake | How to install with Nix or Devbox](https://www.nixhub.io/packages/cmake?utm_source=chatgpt.com)
2. Install lgbm


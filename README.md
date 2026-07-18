# AMME

This project focuses on the **prediction and mechanism analysis of the South China Sea Summer Monsoon (SCSSM) onset date**. It combines machine learning (XGBoost, Random Forest) and traditional statistical methods to analyze how precursor factors influence the interannual and interdecadal variability of SCSSM onset.

## Research Contents

- **Interannual / interdecadal variability of SCSSM onset**: Investigating the trend changes of monsoon onset dates under different PDO phases.
- **Precursor correlation analysis**: Analyzing correlations between SCSSM onset and precursor factors such as ENSO, STT, SAM, LST, and AO, comparing the full period, pre-2010, post-2010, and other sub-periods.
- **Machine learning prediction**: Building a large-scale XGBoost ensemble (500,000 models) and selecting optimal models for year-by-year prediction.
- **Model comparison**: Hindcast and forecast comparison against the SM-17 statistical model and the Hu-25_RF random forest model.
- **Interpretability analysis**: Using SHAP to quantify the year-by-year contributions of each precursor factor to the XGBoost predictions.

## Main Scripts

| Script | Description |
|--------|-------------|
| `XG_model.py` | Train a large-scale XGBoost regression model and save predictions from 500,000 models |
| `predict_base_period_one_year.py` | Select the optimal model year-by-year using a sliding window, and plot prediction schematics |
| `predict_base_period_test_year.py` | Perform model selection and error evaluation for the test period (2016–2025) |
| `predict_base_period_contribute.py` | Use SHAP to analyze the year-by-year and multi-period contributions of each factor |
| `heat_corr.py` | Plot Pearson correlation heatmaps among precursor factors |
| `running_corr_v2.py` | Calculate running correlations between precursor factors and SCSSM onset date |
| `interannual_variability.py` | Analyze PDO, interannual variability, and multi-model prediction error comparison |
| `SM-17/SM-17.py` | Construction and validation of the SM-17 statistical model (STT + ENSO + SAM) |
| `Hu-25_RF/Hu-25_RF_16_25_compare.py` | Training and comparison of the Hu-25 random forest model |

## Data Files

- `precursor_factors.csv`: Yearly series of precursor factors and SCSSM onset dates.
- `lg_scssm.csv`: Predictions from the SM-17 statistical model.
- `predictions_ways_1986_2025.csv`: XG-CE ensemble prediction results.
- `migrated_500000.zip`: Yearly prediction matrix from 500,000 XGBoost models (large file, not committed to Git).
- `Hu-25_RF/yy_pre_rf_model_*.csv`: Yearly prediction results from the random forest model.

## Usage

1. **Data preparation**: Place data files such as `precursor_factors.csv` in the project root directory.
2. **Model training**: Run `XG_model.py` to generate the large-scale model ensemble.
3. **Model selection**: Run `predict_base_period_one_year.py` or `predict_base_period_test_year.py` for year-by-year optimal model selection.
4. **Result analysis**: Run `predict_base_period_contribute.py` to generate SHAP explanation plots; run `heat_corr.py` or `running_corr_v2.py` for correlation analysis.

## Dependencies

Mainly based on the Python scientific computing ecosystem:

- `xarray`, `numpy`, `pandas`, `scipy`
- `matplotlib`, `seaborn`, `cartopy`, `cmaps`, `colormaps`
- `scikit-learn`, `xgboost`, `shap`, `joblib`, `tqdm`
- `eofs`, `sacpy`, `metpy`

## Notes

- Large `.csv` files (e.g., `migrated_500000.csv`) are not included in Git version control. It is recommended to manage them via `.gitignore`.
- For detailed Git workflow and troubleshooting, refer to `git_workflow_guide.md`.

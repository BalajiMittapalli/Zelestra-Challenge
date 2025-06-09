*Note: The notebook currently reads data from `/content/` and saves HPO params to `/content/`. For local execution, you might want to adjust paths or place files accordingly.*

## 🚀 Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/BalajiMittapalli/Zelestra-Challenge.git
    cd Zelestra-Challenge
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Data Placement:**
    *   Download `train.csv` and `test.csv`.
    *   The notebook expects them at `/content/train.csv` and `/content/test.csv`. If running locally, you can:
        *   Create a `data/` directory in the project root and place them there, then update paths in the notebook/script.
        *   Or, if using Colab, upload them to the `/content/` directory of your Colab instance.

## ⚙️ Workflow

The entire machine learning pipeline is executed within the `AWS_Ascends.ipynb` notebook (or its Python script equivalent).

1.  **Environment Setup (Cell 1):** Installs necessary libraries and imports modules. Sets `RANDOM_STATE`.
2.  **Load Data (Cell 2):** Reads `train.csv` and `test.csv`.
3.  **Feature & Target Separation (Cell 3):** Separates target `efficiency`, IDs, and defines initial feature lists.
4.  **Numerical Coercion (Cell 4):** Converts potential object-type numerical columns to numeric, handling errors by coercing to NaN.
5.  **Feature Engineering (Cell 5 & 6):** Creates new features. The `numerical_feats` list is updated.
6.  **Preprocessing Pipelines (Cell 7):** Defines `ColumnTransformer` with separate pipelines for numerical, low-cardinality categorical, and high-cardinality categorical features.
7.  **Hyperparameter Optimization (Cells 8, 9, 10):**
    *   **LightGBM (Cell 8):** Optuna optimizes LGBMRegressor. Best parameters saved to `best_lgb_params.json`.
    *   **CatBoost (Cell 9):** Optuna optimizes CatBoostRegressor (with reduced iterations for HPO speed). Best parameters saved to `best_catboost_hpo_params.json`.
    *   **XGBoost (Cell 10):** Optuna optimizes XGBRegressor (with reduced estimators for HPO speed). Best parameters saved to `best_xgboost_hpo_params_fast_trials.json`.
    *   *Note: HPO can be time-consuming. The number of trials (`n_trials`) and iterations/estimators within the objective functions are set for a balance between performance and speed.*
8.  **Final Model Definitions (Cell 11):**
    *   Loads the saved best HPO parameters from the JSON files.
    *   Defines final base learner pipelines (`lgb_final`, `cat_final`, `xgb_final`) using these parameters and increased `n_estimators`/`iterations` for robust final models.
9.  **Stacking Regressor (Cell 12):** Defines the `StackingRegressor` using the three final base learners and an LGBMRegressor as the meta-learner.
10. **Final Model Training (Cell 13):** Fits the `StackingRegressor` on the full training data.
11. **Prediction & Clipping (Cell 14):**
    *   Makes predictions on the `X_test` data.
    *   Clips predictions to the min/max range of the training target `y` to ensure realistic outputs.
12. **Submission File (Cell 15):** Creates `submission_stacked_enhanced.csv`.
13. **Optuna Summaries (Cell 16):** Prints the best RMSE and parameters found by Optuna for each model.

**To Run:**
*   Open `AWS_Ascends.ipynb` in Jupyter Lab/Notebook or Google Colab.
*   Ensure data files (`train.csv`, `test.csv`) are accessible as per path configurations.
*   Run cells sequentially. The HPO cells (8, 9, 10) can take a significant amount of time.
*   The script will generate JSON files for HPO parameters and a final `submission_stacked_enhanced.csv`.

## 📊 Results

The performance of the models is evaluated using Root Mean Squared Error (RMSE) during cross-validation in the Optuna HPO phase.
*   **LightGBM Best CV RMSE**: ~0.10357 (from HPO logs)
*   **CatBoost Best CV RMSE**: ~0.10290 (from HPO logs with reduced iterations)
*   **XGBoost Best CV RMSE**: ~0.10301 (from HPO logs with reduced estimators)

The final stacked model aims to leverage the diverse strengths of these optimized base learners for improved generalization and predictive accuracy. The final output is `submission_stacked_enhanced.csv`.

## 🔮 Future Improvements

*   **Advanced Feature Engineering**: Explore more complex interactions, time-based features (if applicable from data context not fully evident), or domain-specific features.
*   **Alternative Ensembling**: Experiment with other ensembling techniques like blending or weighted averaging based on CV scores.
*   **Deeper Hyperparameter Optimization**: Increase `n_trials` for Optuna studies or explore different HPO libraries/strategies.
*   **Cross-Validation Strategy**: Investigate more sophisticated CV strategies if data has temporal or group-based dependencies (e.g., TimeSeriesSplit, GroupKFold).
*   **Error Analysis**: Perform a detailed analysis of prediction errors to identify areas where the model struggles and guide further improvements.
*   **Feature Selection**: Implement feature selection techniques to reduce dimensionality and potentially improve model robustness and training speed.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details (if you choose to add one). If no license file is present, it's under standard copyright.

---

Happy coding and may your solar panels always be efficient! 🚀

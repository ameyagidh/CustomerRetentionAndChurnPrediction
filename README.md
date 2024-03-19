# Customer Retention And Churn Prediction

**Overview:**
Customer retention and churn prediction are critical aspects of business management, especially in industries with subscription-based or recurring revenue models. This project aims to develop a predictive model to identify customers who are likely to churn, allowing businesses to take proactive measures to retain them.

## Screenshots
Churn probability

<img width="1143" alt="Screenshot 2024-03-19 at 10 53 09 AM" src="https://github.com/ameyagidh/CustomerRetentionAndChurnPrediction/assets/65457905/b5b3f056-9a58-4d7a-8154-1bb50fdce47e">

**Project Structure:**
1. **Data Loading and Preprocessing:** 
   - Data files are read using the `read_data` function provided in the `data_utils.py` module.
   - Data preprocessing steps include handling missing values, encoding categorical variables, scaling numerical features, and creating additional engineered features.

2. **Feature Engineering:**
   - The `AddFeatures` class in the `feature_eng.py` module adds new engineered features based on the original dataset's columns. These features are designed to capture potential patterns related to customer behavior and demographics.

3. **Model Selection and Training:**
   - A variety of machine learning models are evaluated for their ability to predict customer churn. Models include tree-based classifiers (Random Forest, Extra Trees, LightGBM, XGBoost), k-Nearest Neighbors, and Naive Bayes classifiers.
   - Models are trained using K-fold cross-validation to ensure robust performance assessment.

4. **Model Evaluation:**
   - The `evaluate_models` function in the `model_evaluation.py` module is used to evaluate each model's performance based on a chosen evaluation metric (e.g., recall, accuracy).
   - Evaluation results, including mean scores and standard deviations, are printed for each model.

5. **Pipeline Construction:**
   - Custom pipelines are created for each model using the `make_pipeline` function in the `pipeline_utils.py` module. Pipelines handle data preprocessing, feature engineering, and model training in a streamlined manner.

6. **Hyperparameter Tuning:**
   - Hyperparameters for each model are fine-tuned using grid search or manually specified parameter sets. Tuning is performed to optimize each model's performance further.

7. **Model Deployment:**
   - Once the best-performing model is identified, it can be deployed in a production environment to make real-time predictions on new customer data.
   - Deployment considerations may include model serving infrastructure, monitoring, and periodic model retraining.

**Dependencies:**
- Python 3.x
- Libraries: pandas, numpy, scikit-learn, lightgbm, xgboost

**Usage:**
1. Install the required dependencies using `pip install -r requirements.txt`.
2. Run the main script or Jupyter Notebook to execute the entire pipeline and evaluate different models' performance.
3. Fine-tune hyperparameters if necessary and re-evaluate models to achieve optimal performance.


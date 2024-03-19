import numpy as np
import joblib

def predict(df):
    try:
        # Columns to remove from the DataFrame
        cols_to_remove = ['RowNumber', 'CustomerId']
        
        # Check if 'Exited' column is present and remove it if exists
        if 'Exited' in df.columns:
            df.drop(columns=['Exited'], inplace=True)

        # Remove specified columns if they exist
        for col in cols_to_remove:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        # Load the trained model
        model = joblib.load('../output/final_churn_model_f1_0_45.sav')

        # Predict target probabilities
        test_probs = model.predict_proba(df)[:, 1]

        # Predict target values on test data based on a probability threshold
        test_preds = np.where(test_probs > 0.45, 1, 0)

        # Create a copy of the DataFrame for processing
        test = df.copy()

        # Add predictions and prediction probabilities to the DataFrame
        test['predictions'] = test_preds
        test['pred_probabilities'] = test_probs

        # Extract high churn cases based on a probability threshold
        high_churn_list = test[test.pred_probabilities > 0.7].sort_values(
            by=['pred_probabilities'], ascending=False
        ).reset_index().drop(columns=['index', 'predictions'], axis=1)

        # Print shape and head of high churn list for debugging purposes
        print(high_churn_list.shape)
        print(high_churn_list.head())

        # Return status code 200 (OK) and high churn list
        return 200, high_churn_list

    except Exception as error:
        # Return status code 500 (Internal Server Error) and error message
        return 500, str(error)

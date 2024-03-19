# Import necessary libraries
import os
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template
import pandas as pd
import json

# Load environment variables from .env file
load_dotenv(Path(".env"))

# Load specific environment variables based on environment (prod or dev)
if os.environ.get("ENV", "dev") == "prod":
    load_dotenv(Path(".env.prod"))
if os.environ.get("ENV", "dev") == "dev":
    load_dotenv(Path(".env.dev"))

# Import custom modules
from logging_module import logger
from predictor import predict

# Initialize Flask application
app = Flask(__name__)

# Route for handling input form
@app.route("/", methods=['GET', 'POST'])
def input_form():
    if request.method == 'POST':
        # Get input data from the form
        data = request.form.get('input_data')
        return jsonify({'data': data})  # Return input data as JSON
    return render_template('input_form.html')  # Render input form template

# Route for churn prediction
@app.route("/predict", methods=['POST'])
def churn_prediction():
    logger.debug("Churn Prediction API Called")  # Log API call
    # Convert JSON data to pandas DataFrame
    df = pd.DataFrame(request.json["data"])
    # Perform churn prediction
    status, result = predict(df)
    if status == 200:
        # Convert prediction result to JSON format
        result = json.loads(result.to_json(orient="records"))
        return render_template('churn_prediction.html', result=result)  # Render prediction result template
    else:
        return render_template('churn_prediction.html', errorDetails=result), status  # Render error template if prediction fails

# Entry point of the application
if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask application in debug mode

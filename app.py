from flask import Flask, request, jsonify
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import os

app = Flask(__name__)

# Define a folder to save uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # Check if a file is part of the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        # Check if a file is selected
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Check if the file has a CSV extension
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Invalid file type. Only CSV files are allowed.'}), 400

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file, parse_dates=['Date'])

        # Ensure the 'Demand' column exists
        if 'Demand' not in df.columns:
            return jsonify({'error': 'Demand column not found in the CSV file'}), 400

        # Set the 'Date' column as the index
        df.set_index('Date', inplace=True)

        # Fit the ARIMA model
        model = ARIMA(df['Demand'], order=(1, 1, 1))  # Adjust the order as needed
        model_fit = model.fit()

        # Make predictions (forecasting the next 10 periods)
        forecast = model_fit.forecast(steps=10)

        # Prepare the response
        forecast_data = forecast.tolist()  # Convert forecast result to list
        return jsonify({'forecast': forecast_data}), 200

    except pd.errors.EmptyDataError:
        return jsonify({'error': 'The CSV file is empty or corrupted'}), 400

    except pd.errors.ParserError:
        return jsonify({'error': 'Error parsing CSV file'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)

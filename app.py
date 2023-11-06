from flask import Flask, jsonify, render_template
import requests
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = load_model('stock_prediction_model.h5')

# Load the historical stock data
api_key = 'DCULDMORQ5N6BENV'
symbol = 'AAPL'
function = 'TIME_SERIES_DAILY'
url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}"
response = requests.get(url)
data = response.json()
daily_data = data['Time Series (Daily)']
df = pd.DataFrame(daily_data).T
df.index = pd.to_datetime(df.index)
df = df.sort_index()
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# Data Preprocessing
scaler = MinMaxScaler()
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Define the number of previous days' data to use for prediction
look_back = 10

# Create sequences of data for prediction
X = []
for i in range(len(df) - look_back):
    X.append(df['Close'][i:i + look_back])
X = np.array(X)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    # Predict the next day's closing price
    last_sequence = X[-1]
    last_sequence = np.reshape(last_sequence, (1, look_back, 1))
    predicted_value = model.predict(last_sequence)
    predicted_value = float(predicted_value)  # Convert to Python float
    return jsonify({'predicted_price': predicted_value})

if __name__ == '__main__':
    app.run(debug=True)

#!/usr/bin/env python
# coding: utf-8

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Set plot style
plt.style.use("fivethirtyeight")

# Fetch stock data
end = datetime.now()
start = datetime(end.year - 15, end.month, end.day)
stock = 'BTC-USD'
stock_data = yf.download(stock, start=start, end=end)

# Display data insights
print("Head of stock data:")
print(stock_data.head())

print("\nTail of stock data:")
print(stock_data.tail())

print("\nSummary statistics:")
print(stock_data.describe().T)

print("\nData info:")
stock_data.info()

# Extract close price data
closing_price = stock_data[['Close']]

# Plot close price over time
plt.figure(figsize=(15, 6))
plt.plot(closing_price.index, closing_price['Close'], label='Close Price', color='blue', linewidth=2)
plt.title("Close price of Bitcoin over time", fontsize=16)
plt.xlabel("Years", fontsize=14)
plt.ylabel("Close Price", fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.show()

# Calculate moving averages
closing_price['MA_365'] = closing_price['Close'].rolling(window=365).mean()
closing_price['MA_100'] = closing_price['Close'].rolling(window=100).mean()

# Plot close price with moving averages
plt.figure(figsize=(15, 6))
plt.plot(closing_price.index, closing_price['Close'], label='Close Price', color='blue', linewidth=2)
plt.plot(closing_price.index, closing_price['MA_365'], label='365 Moving Average', color='red', linestyle="--", linewidth=2)
plt.plot(closing_price.index, closing_price['MA_100'], label='100 Moving Average', color='green', linestyle="--", linewidth=2)
plt.title("Close price with moving averages", fontsize=16)
plt.xlabel("Years", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.show()

# Scale data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_price[['Close']].dropna())

# Prepare data for LSTM model
x_data = []
y_data = []
base_days = 100
for i in range(base_days, len(scaled_data)):
    x_data.append(scaled_data[i - base_days:i])
    y_data.append(scaled_data[i])

x_data = np.array(x_data)
y_data = np.array(y_data)

# Split into train and test sets
train_size = int(len(x_data) * 0.9)
x_train, y_train = x_data[:train_size], y_data[:train_size]
x_test, y_test = x_data[train_size:], y_data[train_size:]

# Define LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(64, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

# Train model
model.fit(x_train, y_train, batch_size=5, epochs=10)

# Predict test data
predictions = model.predict(x_test)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_test)

# Plot predictions vs actual data
plotting_data = pd.DataFrame({
    'Original': inv_y_test.flatten(),
    'Prediction': inv_predictions.flatten()
}, index=closing_price.index[train_size + base_days:])

plt.figure(figsize=(15, 6))
plt.plot(plotting_data.index, plotting_data['Original'], label='Original', color='blue', linewidth=2)
plt.plot(plotting_data.index, plotting_data['Prediction'], label='Prediction', color='red', linewidth=2)
plt.title("Prediction vs Actual Close Price", fontsize=16)
plt.xlabel("Years", fontsize=14)
plt.ylabel("Close Price", fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.show()

# Predict future prices
last_100 = scaled_data[-100:].reshape(1, -1, 1)
future_predictions = []
for _ in range(10):
    next_days = model.predict(last_100)
    future_predictions.append(scaler.inverse_transform(next_days))
    last_100 = np.append(last_100[:, 1:, :], next_days.reshape(1, 1, -1), axis=1)

# Flatten predictions
future_predictions = np.array(future_predictions).flatten()

# Plot future predictions
plt.figure(figsize=(15, 6))
plt.plot(range(1, 11), future_predictions, marker="o", label='Prediction of Future Prices', color='purple', linewidth=2)

for i, val in enumerate(future_predictions):
    plt.text(i + 1, val, f'{val:.2f}', fontsize=10, ha='center', va='bottom', color='black')

plt.title("Future Close Prices for 10 Days", fontsize=16)
plt.xlabel("Day Ahead", fontsize=14)
plt.ylabel("Close Price", fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.show()

# Save the model
model.save("model.keras")

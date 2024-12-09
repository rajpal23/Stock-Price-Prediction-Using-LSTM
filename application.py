import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Set Seaborn style
sns.set(style="whitegrid")

# Load the pre-trained model
model = load_model(r'C:\Users\Dell\Desktop\My work\Finall projects\Stock Price prediction\New folder\Stock Predictions Model.keras')

# Streamlit header
st.header('Stock Market Predictor')

# User input for stock symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# Download stock data
data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

# Split data into train and test sets
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Moving averages
ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

# Price vs MA50
st.subheader('Price vs MA50')
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.lineplot(x=data.index, y=ma_50_days, color='red', label='MA50', ax=ax1)
sns.lineplot(x=data.index, y=data.Close, color='green', label='Close Price', ax=ax1)
ax1.set_title('Price vs MA50')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
st.pyplot(fig1)

# Price vs MA50 vs MA100
st.subheader('Price vs MA50 vs MA100')
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.lineplot(x=data.index, y=ma_50_days, color='red', label='MA50', ax=ax2)
sns.lineplot(x=data.index, y=ma_100_days, color='blue', label='MA100', ax=ax2)
sns.lineplot(x=data.index, y=data.Close, color='green', label='Close Price', ax=ax2)
ax2.set_title('Price vs MA50 vs MA100')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
st.pyplot(fig2)

# Price vs MA100 vs MA200
st.subheader('Price vs MA100 vs MA200')
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.lineplot(x=data.index, y=ma_100_days, color='red', label='MA100', ax=ax3)
sns.lineplot(x=data.index, y=ma_200_days, color='blue', label='MA200', ax=ax3)
sns.lineplot(x=data.index, y=data.Close, color='green', label='Close Price', ax=ax3)
ax3.set_title('Price vs MA100 vs MA200')
ax3.set_xlabel('Date')
ax3.set_ylabel('Price')
st.pyplot(fig3)

# Prepare data for prediction
x, y = [], []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Make predictions
predict = model.predict(x)

# Scale predictions back to original
scale = 1 / scaler.scale_
predict = predict * scale
y = y * scale

# Original Price vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.lineplot(x=range(len(predict)), y=predict.flatten(), color='red', label='Predicted Price', ax=ax4)
sns.lineplot(x=range(len(y)), y=y.flatten(), color='green', label='Original Price', ax=ax4)
ax4.set_title('Original Price vs Predicted Price')
ax4.set_xlabel('Time')
ax4.set_ylabel('Price')
ax4.legend()
st.pyplot(fig4)

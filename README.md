# Stock-Price-Prediction-Using-LSTM

📋 Project Summary
This project utilizes Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network (RNN), to predict stock prices based on historical data. By capturing time-series patterns, the model forecasts future stock trends with a focus on accuracy and visualization.

🚀 Features
Time-Series Analysis: Processes historical stock data to identify trends.
LSTM Neural Network: Employs deep learning to predict future stock prices.
Data Visualization: Graphical comparison of predicted vs. actual prices.
Model Deployment: Saved and reusable for real-time predictions.
🛠️ Key Steps
1. Data Collection and Preprocessing
Collected historical stock price data using financial APIs (e.g., Yahoo Finance).
Normalized data for consistency and created input-output sequences for the model.
2. Model Development
Built a Sequential LSTM model with:
Multiple LSTM layers for time-series learning.
Dropout layers to prevent overfitting.
Compiled the model with suitable loss functions and optimizers.
3. Training and Testing
Trained on historical data split into training and test sets.
Evaluated performance using metrics like Mean Squared Error (MSE).
4. Prediction and Evaluation
Predicted future stock prices using the trained model.
Visualized predicted vs. actual stock prices for performance assessment.
5. Deployment
Saved the trained model (model.save()), enabling reuse without retraining.
📊 Results
Accuracy: Achieved reliable predictions with low error rates.
Visual Insights: Generated intuitive plots comparing predictions to actual values.

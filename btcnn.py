import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import time
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Constants & Color Setup ---
INFO_COLOR = "\033[96m"
ERROR_COLOR = "\033[91m"
WARNING_COLOR = "\033[93m"
RESET_COLOR = "\033[0m"

# --- Clear Console & Display ASCII Art ---
os.system('cls' if os.name == 'nt' else 'clear')

ascii_art = f"""
{INFO_COLOR}
██████╗ ████████╗ ██████╗    ███╗   ██╗███╗   ██╗
██╔══██╗╚══██╔══╝██╔════╝    ████╗  ██║████╗  ██║
██████╔╝   ██║   ██║         ██╔██╗ ██║██╔██╗ ██║
██╔══██╗   ██║   ██║         ██║╚██╗██║██║╚██╗██║
██████╔╝   ██║   ╚██████╗    ██║ ╚████║██║ ╚████║
╚═════╝    ╚═╝    ╚═════╝    ╚═╝  ╚═══╝╚═╝  ╚═══╝
              Bitcoin Neural Network
{RESET_COLOR}
"""
print(ascii_art)

# --- Loading Animation for Training ---
def loading_animation(message="Processing"):
    animation = "|/-\\"
    for i in range(20):  # Adjust for desired duration
        sys.stdout.write(f"\r{INFO_COLOR}{message}... {animation[i % len(animation)]}{RESET_COLOR}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the line

# --- Mode Selection ---
def get_mode():
    mode = input(f"{INFO_COLOR}Enter mode (1: Training Mode, 2: Prediction Mode, 3: Verbose Info Mode): {RESET_COLOR}").strip()
    return mode

# --- Data Fetching & Preprocessing ---
def fetch_data():
    print(f"{INFO_COLOR}Fetching BTC-USD data for the last 5 days...{RESET_COLOR}")
    try:
        btc_data = yf.download(tickers="BTC-USD", interval="1m", period="5d")
        btc_data = btc_data[['Close']].ffill()  # Use ffill to fill missing values
    except Exception as e:
        print(f"{ERROR_COLOR}Error fetching data: {e}{RESET_COLOR}")
        sys.exit(1)
    
    # Exclude the last 30 minutes
    lookahead_minutes = 30
    btc_data = btc_data[:-lookahead_minutes]
    
    return btc_data

# --- Data Normalization ---
def normalize_data(btc_data):
    scaler_close = StandardScaler()
    
    if 'Close' in btc_data.columns:
        btc_data['Normalized_Close'] = scaler_close.fit_transform(btc_data[['Close']])
    else:
        print(f"{ERROR_COLOR}Error: 'Close' column is missing in the dataset!{RESET_COLOR}")
        sys.exit(1)
    
    return btc_data, scaler_close

# --- Feature Creation (Sliding Window) ---
def create_sliding_window(data, window_size, lookahead):
    features, targets = [], []
    for i in range(len(data) - window_size - lookahead):
        features.append(data[i:i+window_size])
        targets.append(data[i+window_size+lookahead-1])
    return np.array(features), np.array(targets)

# --- Neural Network Definition ---
class BTCPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=4, dropout=0.2):
        super(BTCPricePredictor, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# --- Mode 1: Training Mode ---
def train_model(X_train, y_train, model, optimizer, criterion, epochs=10):
    training_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    return training_losses

# --- Mode 2: Prediction Mode ---
def predict_model(X_val, y_val, model, scaler_close, btc_data):
    model.eval()
    
    # Animation while predicting
    print(f"{INFO_COLOR}Making Predictions...{RESET_COLOR}")
    loading_animation(message="Predicting")

    with torch.no_grad():
        predictions = model(X_val).squeeze().numpy()

    # Plot predicted prices
    plt.figure(figsize=(10, 6))

    # Get time indices for the validation set predictions
    predicted_time = btc_data.index[-len(predictions):]

    # Convert time to minutes for better clarity (using relative time)
    time_in_minutes = [(t - predicted_time[0]).total_seconds() / 60 for t in predicted_time]

    # Inverse-transform the predictions to actual price scale
    predicted_prices = scaler_close.inverse_transform(predictions.reshape(-1, 1)).squeeze()

    # Plot the predicted prices
    plt.plot(time_in_minutes, predicted_prices, label="Predicted Prices", color="red")
    plt.title(f"BTC-USD Predicted Prices (30 min lookahead)")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

    # Display Actual vs Predicted values for the first 10 predictions
    print(f"{INFO_COLOR}Displaying actual vs predicted values:{RESET_COLOR}")
    for i in range(10):
        actual_price = scaler_close.inverse_transform([[y_val[i].item()]])[0][0]
        predicted_price = scaler_close.inverse_transform([[predictions[i]]])[0][0]
        print(f"Actual: {actual_price:.2f}, Predicted: {predicted_price:.2f}")

# --- Mode 3: Verbose Info Mode ---
def verbose_info(X_train, X_val, model, training_losses):
    print(f"{INFO_COLOR}Verbose Mode: Displaying additional information...{RESET_COLOR}")
    
    print(f"{INFO_COLOR}Data Preview (First 10 rows):{RESET_COLOR}")
    print(btc_data.head(10))
    
    print(f"{INFO_COLOR}\nData Statistics:{RESET_COLOR}")
    print(btc_data.describe())
    
    print(f"{INFO_COLOR}\nNeural Network Architecture:{RESET_COLOR}")
    print(model)

    print(f"{INFO_COLOR}\nTraining Data Shape:{RESET_COLOR}")
    print(X_train.shape)

    print(f"{INFO_COLOR}\nValidation Data Shape:{RESET_COLOR}")
    print(X_val.shape)

    # Display training loss chart
    print(f"{INFO_COLOR}\nTraining Loss Chart:{RESET_COLOR}")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_losses)+1), training_losses, label="Training Loss", color="blue")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Main Execution Flow ---
def main():
    mode = get_mode()

    # Fetch and preprocess data
    btc_data = fetch_data()
    btc_data, scaler_close = normalize_data(btc_data)
    
    # Prepare features and targets
    window_size = 30
    lookahead_minutes = 30
    btc_data_for_training = btc_data[:-lookahead_minutes]
    features, targets = create_sliding_window(btc_data_for_training['Normalized_Close'].values, window_size, lookahead_minutes)

    # Split data
    validation_split = 0.2
    X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=validation_split, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Initialize model, criterion, and optimizer
    model = BTCPricePredictor(input_size=window_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if mode == '1':  # Training Mode
        epochs = int(input(f"{INFO_COLOR}Enter epochs: {RESET_COLOR}"))
        training_losses = train_model(X_train, y_train, model, optimizer, criterion, epochs)

        # Plot Training Loss
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs+1), training_losses, label="Training Loss", color="blue")
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Save the trained model
        torch.save(model.state_dict(), "model.pth")
    
    elif mode == '2':  # Prediction Mode
        model.load_state_dict(torch.load("model.pth"))
        predict_model(X_val, y_val, model, scaler_close, btc_data)
    
    elif mode == '3':  # Verbose Info Mode
        verbose_info(X_train, X_val, model, training_losses)

if __name__ == "__main__":
    main()

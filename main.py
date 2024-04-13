import streamlit as st
from pycoingecko import CoinGeckoAPI
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# Function to calculate simple moving averages (SMA)
def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

# Function to calculate Relative Strength Index (RSI)
def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to get historical data and calculate technical indicators
def get_crypto_data(crypto_symbol):
    try:
        # Fetch historical data from Yahoo Finance for the past 30 days
        crypto_data = yf.download(crypto_symbol, period="1mo", interval="1d")
        
        # Calculate technical indicators
        crypto_data['SMA_20'] = calculate_sma(crypto_data, 20)
        crypto_data['SMA_50'] = calculate_sma(crypto_data, 50)
        crypto_data['RSI'] = calculate_rsi(crypto_data, 14)
        
        return crypto_data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Function to fetch crypto score from CoinGecko API
def get_crypto_score(crypto_id):
    cg = CoinGeckoAPI()

    if crypto_id:
        crypto_data = cg.get_coin_by_id(crypto_id)
        
        if crypto_data:
            # You can adjust these criteria based on your analysis
            market_score = crypto_data['market_cap_rank']  # Example criterion
            community_score = crypto_data['community_score'] if 'community_score' in crypto_data else 0  # Example criterion
            development_score = crypto_data['developer_score'] if 'developer_score' in crypto_data else 0  # Example criterion
            
            # Calculate total score
            total_score = (market_score + community_score + development_score) / 3
            
            # Normalize score to a scale of 100
            normalized_score = (total_score) * 100
            
            return normalized_score
        else:
            return None
    else:
        return None 

# Function to fetch news articles related to the cryptocurrency
def fetch_news(crypto_name):
    url = f"https://newsapi.org/v2/everything?q={crypto_name}&apiKey=6e71d7856bcb4c2d9ddef42386e1f99f"
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        return news_data['articles']
    else:
        return None

# Function to perform sentiment analysis on news articles
def perform_sentiment_analysis(articles):
    sentiments = []
    for article in articles:
        if article['title'] and article['description']:
            text = article['title'] + " " + article['description']
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            sentiments.append(polarity)
    return sentiments

# Function to train a simple linear regression model for price prediction
def train_price_prediction_model(data):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Close'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to predict future prices using the trained model
def predict_prices(model, data, num_days):
    last_index = len(model.predict(np.arange(len(data)).reshape(-1, 1))) - 1
    next_days = np.arange(last_index + 1, last_index + num_days + 1).reshape(-1, 1)
    return model.predict(next_days)


# Main function to create the web app
def main():
    # Set title and description
    st.title("Cryptocurrency Analysis")
    st.write("Enter the name of the cryptocurrency to see its analysis.")
    
    # Mapping between common names and ticker symbols
    crypto_name_to_symbol = {
        "bitcoin": "BTC-USD",
        "ethereum": "ETH-USD",
        "litecoin": "LTC-USD",
        # Add more mappings as needed
    }
    
    # Get user input for cryptocurrency name
    crypto_name = st.text_input("Enter the name of the cryptocurrency (e.g., Bitcoin):")
    
    # Check if the user has input a cryptocurrency name
    if crypto_name:
        # Convert the name to symbol
        crypto_symbol = crypto_name_to_symbol.get(crypto_name.lower())
        
        if crypto_symbol:
            # Call get_crypto_data function to get the data
            crypto_data = get_crypto_data(crypto_symbol)
            
            # Check if the data is available
            if crypto_data is not None:
                # Display fundamental analysis (historical data)
                st.subheader("Fundamental Analysis (Past 30 Days)")
                
                # Plot Open, High, Low, Close prices in one chart with different line styles
                st.write("Price (Open, High, Low, Close):")
                price_chart = crypto_data[['Open', 'High', 'Low', 'Close']].plot(
                    figsize=(12, 6),
                    style=['--', '-', '--', '-'],
                    color='blue'
                )
                st.pyplot(price_chart.figure)
                
                st.write("Volume:")
                st.line_chart(crypto_data['Volume'])
                
                # Display technical analysis (SMA and RSI)
                st.subheader("Technical Analysis")
                st.write("Simple Moving Averages (SMA):")
                st.line_chart(crypto_data[['Close', 'SMA_20', 'SMA_50']])
                st.write("Relative Strength Index (RSI):")
                st.line_chart(crypto_data['RSI'])
                
                # Display crypto score
                score = get_crypto_score(crypto_name.lower())
                if score is not None:
                    st.subheader("Crypto Score")
                    st.write(f"The score to buy {crypto_name} is:")
                    st.progress(score / 100)
                    st.write(score)
                else:
                    st.write("Unable to fetch crypto score.")
                model = train_price_prediction_model(crypto_data)
                
                # Predict future prices for the next 7 days
                num_days = 7
                predicted_prices = predict_prices(model, crypto_data, num_days)
                
                # Display price prediction chart
                st.subheader("Price Prediction for Next 7 Days")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=np.arange(len(crypto_data)), y=crypto_data['Close'], mode='lines', name='Actual Price'))
                fig.add_trace(go.Scatter(x=np.arange(len(crypto_data), len(crypto_data) + num_days), y=predicted_prices, mode='lines', name='Predicted Price'))
                st.plotly_chart(fig)
                # Fetch news articles related to the cryptocurrency
                articles = fetch_news(crypto_name)
                
                if articles:
                    # Perform sentiment analysis on news articles
                    sentiments = perform_sentiment_analysis(articles)
                    st.subheader("Overall Sentiments")
                    sent=sum(sentiments)/len(sentiments)
                    st.progress(sent)
                    st.write(sent)
                    # Display news headlines and sentiments
                    st.subheader("News Sentiments")
                    for i, article in enumerate(articles):
                        if i<len(sentiments):
                            st.write(f"{i+1}. {article['title']}")
                            st.write(f"Description: {article['description']}")
                            st.write(f"Sentiment: {sentiments[i]}")
                            st.write("---")
                        else:
                            break
                else:
                    st.write("No news articles found.")
                
                
                
            else:
                # Display error message if data fetching failed
                st.write("Error fetching cryptocurrency data. Please try again later.")
        else:
            # Display error message if the cryptocurrency name is not found
            st.write("Cryptocurrency not found. Please enter a valid name.")
    
# Run the main function to start the web app
if __name__ == "__main__":
    main()

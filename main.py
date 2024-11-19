import os
import requests
import json
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wallet import get_exchange_rate, execute_trade 
from threading import Thread
from ttkthemes import ThemedStyle
from PIL import Image, ImageTk
from plyer import notification
import webbrowser
import queue

# Tkinter 
matplotlib.use('TkAgg')

#  API STUFF

# CoinMarketCap API Setup
CMC_API_KEY = "-"  # Replace with your actual API key (REMOVED CAUSE PUBLIC)
CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"

# NewsAPI.org Setup
NEWSAPI_KEY = "-"  # Replace with your actual API key (REMOVED CAUSE PUBLIC)
NEWSAPI_URL = "https://newsapi.org/v2/everything"

#  Utility stuff


def send_notification(title, message):
    """
    Send desktop notification using plyer.
    """
    try:
        notification.notify(
            title=title,
            message=message,
            app_icon=None, # did not work, leave blank
            timeout=5
        )
    except Exception as e:
        print(f"Notification Error: {e}")


def export_data(dataframe, filename='export.csv'):
    """
    Export DataFrame to CSV.
    """
    try:
        if dataframe.empty:
            messagebox.showwarning("Export Warning", "No data available to export.")
            send_notification("Export Warning", "No data available to export.")
            return
        dataframe.to_csv(filename, index=False)
        messagebox.showinfo("Export Successful", f"Data exported to {filename} successfully!")
        send_notification("Export Successful", f"Data exported to {filename} successfully!")
    except Exception as e:
        messagebox.showerror("Export Failed", f"An error occurred: {e}")
        send_notification("Export Failed", f"An error occurred: {e}")


def export_chart(fig, filename='chart.png'):
    """
    Export matplotlib figure to PNG.
    """
    try:
        if fig is None:
            messagebox.showwarning("Export Warning", "No chart available to export.")
            send_notification("Export Warning", "No chart available to export.")
            return
        fig.savefig(filename)
        messagebox.showinfo("Export Successful", f"Chart exported to {filename} successfully!")
        send_notification("Export Successful", f"Chart exported to {filename} successfully!")
    except Exception as e:
        messagebox.showerror("Export Failed", f"An error occurred: {e}")
        send_notification("Export Failed", f"An error occurred: {e}")


def calculate_trend_indicators(data):
    """
    Calculate trend indicators like Moving Averages and Bollinger Bands.
    """
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    # Bollinger Bands
    data['STD50'] = data['Close'].rolling(window=50).std()
    data['Upper_BB'] = data['MA50'] + (data['STD50'] * 2)
    data['Lower_BB'] = data['MA50'] - (data['STD50'] * 2)

    return data


#  Sentiment Analysis Functions 


def fetch_market_news(query="cryptocurrency", language="en", page_size=100):
    """
    Fetches recent news articles related to the given query using NewsAPI.

    Args:
        query (str): The search query (default is "cryptocurrency").
        language (str): Language of the news articles (default is English).
        page_size (int): Number of articles to fetch (max 100).

    Returns:
        list: A list of news articles.
    """
    url = NEWSAPI_URL
    headers = {
        'Authorization': f'Bearer {NEWSAPI_KEY}',
        'Accepts': 'application/json'
    }
    params = {
        'q': query,
        'language': language,
        'pageSize': page_size,
        'sortBy': 'publishedAt'
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])
        return articles
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while fetching news: {http_err}")
    except Exception as e:
        print(f"Error fetching news: {e}")
    return []


def categorize_sentiment(compound_score):
    """
    Categorize sentiment based on compound score.

    Args:
        compound_score (float): Compound sentiment score from VADER.

    Returns:
        str: 'Positive', 'Negative', or 'Neutral'
    """
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def analyze_sentiment(articles):
    """
    Analyzes the sentiment of a list of news articles.

    Args:
        articles (list): A list of news articles.

    Returns:
        pd.DataFrame: DataFrame containing publication date and sentiment scores.
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []
    for article in articles:
        # Prefer 'description' over 'content', then fallback to 'title'
        content = article.get('description') or article.get('content') or article.get('title') or ""
        if not content:
            continue
        sentiment = analyzer.polarity_scores(content)
        sentiment_scores.append({
            'publishedAt': article.get('publishedAt'),
            'compound': sentiment['compound']
        })

    sentiment_df = pd.DataFrame(sentiment_scores)
    if sentiment_df.empty:
        return pd.DataFrame({
            'publishedAt': [],
            'compound': []
        })

    # Convert 'publishedAt' to datetime
    sentiment_df['publishedAt'] = pd.to_datetime(sentiment_df['publishedAt'])

    # Sort by date
    sentiment_df.sort_values('publishedAt', inplace=True)

    return sentiment_df


def get_all_crypto_symbols():
    """
    Fetches the top 5 cryptocurrency symbols by market capitalization from CoinMarketCap.

    Returns:
        list: A list of top 5 cryptocurrency symbols.
    """
    try:
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        headers = {
            'X-CMC_PRO_API_KEY': CMC_API_KEY,
            'Accepts': 'application/json'
        }
        params = {
            'start': '1',
            'limit': '5',  # Limited to top 5
            'convert': 'USD'
        }
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        symbols = [item['symbol'] for item in data['data'][:5]]  # Ensure only top 5
        return symbols
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while fetching crypto symbols: {http_err}")
    except Exception as e:
        print(f"Error fetching crypto symbols: {e}")
    return ["BTC", "ETH", "LTC", "XRP", "BNB"]  # Fallback to top 5


#  Helper Functions 


def fetch_and_prepare_historical_data(symbol, interval):
    """
    Fetch and prepare historical data for a given symbol and interval.
    """
    try:
        # Determine period based on interval for better data granularity
        if interval.endswith('h'):
            period = '90d'  # 3 months for hourly data
        elif interval.endswith('wk'):
            period = '5y'  # 5 years for weekly data
        else:
            period = '5y'  # 5 years for daily data

        # Fetch data using yfinance
        data = yf.download(tickers=symbol, period=period, interval=interval)
        if data.empty:
            print(f"No data fetched for {symbol}.")
            return pd.DataFrame()
        data.reset_index(inplace=True)

        # Keep 'Date' as datetime
        if 'Date' not in data.columns and 'Datetime' in data.columns:
            data.rename(columns={'Datetime': 'Date'}, inplace=True)
        elif 'Date' not in data.columns and 'Timestamp' in data.columns:
            data.rename(columns={'Timestamp': 'Date'}, inplace=True)

        return data
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()


def fetch_news_sentiment(query, from_date, to_date):
    """
    Fetch news articles and calculate average sentiment.

    Args:
        query (str): Search query.
        from_date (str): Start date in YYYY-MM-DD format.
        to_date (str): End date in YYYY-MM-DD format.

    Returns:
        tuple: (average_sentiment, list_of_articles)
    """
    try:
        articles = fetch_market_news(query=query, language="en", page_size=100)
        if not articles:
            return 0.0, []

        sentiment_df = analyze_sentiment(articles)
        sentiments = sentiment_df['compound'].tolist()

        if sentiments:
            average_sentiment = average_sentiment_percentage(sum(sentiments) / len(sentiments))
            return average_sentiment, articles
        else:
            return 0.0, []
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while fetching news sentiment: {http_err}")
    except Exception as e:
        print(f"Error fetching news sentiment: {e}")
    return 0.0, []


def build_and_train_lstm(data, forecast_horizon, include_sentiment=False):
    """
    Build and train an LSTM model for price prediction.
    Returns model, training data, testing data, and future data.
    """
    # Ensure 'Close' column exists
    if 'Close' not in data.columns:
        raise KeyError("DataFrame must contain 'Close' column.")
    if 'Date' not in data.columns:
        raise KeyError("DataFrame must contain 'Date' column.")

    # Preprocessing
    scaler_close = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler_close.fit_transform(data[['Close']])

    scaler_sentiment = None
    scaled_sentiment = None
    if include_sentiment:
        scaler_sentiment = MinMaxScaler(feature_range=(0, 1))
        scaled_sentiment = scaler_sentiment.fit_transform(data[['Sentiment']])
        scaled_data = np.concatenate([scaled_close, scaled_sentiment], axis=1)
        features = ['Close', 'Sentiment']
    else:
        scaled_data = scaled_close
        features = ['Close']

    # Create training and testing datasets
    training_data_len = int(np.ceil(0.8 * len(scaled_data)))

    train_data = scaled_data[:training_data_len]
    test_data = scaled_data[training_data_len - 60:]  # Include some data for prediction

    # Create the training dataset
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i])
        y_train.append(train_data[i, 0])  # Predicting 'Close' price

    # Convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1)

    # Create the testing dataset
    x_test, y_test = [], data['Close'].values[training_data_len:]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Get model predictions
    predictions = model.predict(x_test)
    predictions = scaler_close.inverse_transform(predictions)  # Inverse transform only 'Close'

    # Inverse transform y_test
    y_test_inv = scaler_close.inverse_transform(y_test.reshape(-1, 1))

    # Prepare DataFrames for plotting
    train_data_df = data.iloc[:training_data_len].copy()
    train_data_df = train_data_df.reset_index(drop=True)

    test_data_df = data.iloc[training_data_len:].copy()
    test_data_df.reset_index(drop=True, inplace=True)
    test_data_df['Predicted_Close'] = predictions.flatten()

    # Future Predictions
    future_dates = pd.date_range(start=data['Date'].iloc[-1] + timedelta(days=1), periods=forecast_horizon, freq='D')

    last_sequence = scaled_data[-60:]
    current_sequence = last_sequence.reshape(1, 60, len(features))
    future_pred = []

    # Calculate average sentiment for future predictions if included
    average_sentiment = data['Sentiment'].mean() if include_sentiment else None

    for _ in range(forecast_horizon):
        pred = model.predict(current_sequence)[0][0]
        future_pred.append(pred)
        if include_sentiment and average_sentiment is not None:
            # Append the predicted Close and average Sentiment
            new_entry = np.array([pred, average_sentiment]).reshape(1, 1, 2)
            current_sequence = np.concatenate((current_sequence[:, 1:, :], new_entry), axis=1)
        elif not include_sentiment:
            # Append only the predicted Close
            new_entry = np.array([pred]).reshape(1, 1, 1)
            current_sequence = np.concatenate((current_sequence[:, 1:, :], new_entry), axis=1)
        else:
            # If sentiment is not included, append only the Close
            new_entry = np.array([pred]).reshape(1, 1, 1)
            current_sequence = np.concatenate((current_sequence[:, 1:, :], new_entry), axis=1)

    # Inverse transform future predictions
    future_pred_inv = scaler_close.inverse_transform(np.array(future_pred).reshape(-1, 1))

    future_data_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_pred_inv.flatten()
    })

    return model, train_data_df, test_data_df, future_data_df


# ----- Class-Based Application =


class CryptoDashboard:
    def __init__(self, root, average_sentiment=None):
        self.root = root
        self.root.title("Crypto Dashboard")
        self.root.geometry("1200x800")  # Adjusted window size
        self.root.resizable(True, True)

        # Apply themed style
        self.style = ThemedStyle(root)
        self.style.set_theme("arc")  # Choose a base theme from ttkthemes

        # Configure custom styles
        self.style.configure('Header.TLabel', font=('Helvetica', 18, 'bold'), foreground='#00ff00')
        self.style.configure('SubHeader.TLabel', font=('Helvetica', 14, 'bold'), foreground='#FFD700')
        self.style.configure('TButton', font=('Helvetica', 12), padding=10)
        self.style.configure('TFrame', background='#2E2E2E')
        self.style.configure('TNotebook.Tab', font=('Helvetica', 12, 'bold'))
        self.style.configure('TLabel', background='#2E2E2E', foreground='white')
        self.style.configure('Status.TLabel', font=('Helvetica', 10), background='#2E2E2E', foreground='white')
        self.style.configure('Positive.TLabel', foreground='green')
        self.style.configure('Negative.TLabel', foreground='red')
        self.style.configure('Neutral.TLabel', foreground='yellow')

        # Initialize variables
        self.cryptocurrency_symbols = get_all_crypto_symbols()  # Fetch top 5 symbols
        if not self.cryptocurrency_symbols:
            self.cryptocurrency_symbols = ["BTC", "ETH", "LTC", "XRP", "BNB"]  # Fallback to top 5

        self.price_vars = {symbol: tk.StringVar(value="Fetching...") for symbol in self.cryptocurrency_symbols}
        self.trend_vars = {symbol: tk.StringVar(value="MA50: N/A | MA200: N/A") for symbol in self.cryptocurrency_symbols}
        self.sentiment_var_home = tk.StringVar(
            value=f"{average_sentiment_percentage(average_sentiment):.2f}%" if average_sentiment else "N/A")

        # Greeting Variable
        self.greeting_var = tk.StringVar(value="Hello User")

        # Queue for thread-safe communication
        self.queue = queue.Queue()

        # Create Notebook
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.home_frame = ttk.Frame(self.notebook)
        self.prediction_frame = ttk.Frame(self.notebook)
        self.wallet_frame = ttk.Frame(self.notebook)
        self.info_frame = ttk.Frame(self.notebook)
        self.settings_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.home_frame, text='Home')
        self.notebook.add(self.prediction_frame, text='Predictions')
        self.notebook.add(self.wallet_frame, text='Wallet')
        self.notebook.add(self.info_frame, text='Info')
        self.notebook.add(self.settings_frame, text='Settings')

        # Setup Home Tab
        self.setup_home_tab()

        # Setup Predictions Tab
        self.setup_predictions_tab()

        # Setup Wallet Tab
        self.setup_wallet_tab()

        # Setup Info Tab
        self.setup_info_tab()

        # Setup Settings Tab
        self.setup_settings_tab()

        # Setup Status Bar
        self.setup_status_bar()

        # Start real-time updates
        self.update_live_prices()
        self.update_trend_indicators()
        self.update_market_sentiment()

    def setup_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_var.set("Application started.")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor='w',
                                    style='Status.TLabel')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_home_tab(self):
        # Header
        ttk.Label(self.home_frame, text="Crypto Dashboard", style='Header.TLabel').pack(pady=20)

        # Greeting Label (After login)
        greeting_label = ttk.Label(self.home_frame, textvariable=self.greeting_var, style='SubHeader.TLabel')
        greeting_label.pack(pady=10)

        # Grid Frame for Crypto Tickers
        grid_frame = ttk.Frame(self.home_frame, padding=10)
        grid_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        # Configure grid columns
        for i in range(len(self.cryptocurrency_symbols)):
            grid_frame.columnconfigure(i, weight=1)

        self.crypto_frames = {}  # To store frames for updates

        for idx, symbol in enumerate(self.cryptocurrency_symbols):
            frame = ttk.Frame(grid_frame, padding=10, relief=tk.RIDGE)
            frame.grid(row=0, column=idx, padx=10, pady=10, sticky='nsew')
            self.crypto_frames[symbol] = frame

            # Ticker Label
            ttk.Label(frame, text=symbol, font=('Helvetica', 14, 'bold')).pack(pady=5)

            # Price Label
            price_label = ttk.Label(frame, textvariable=self.price_vars[symbol], font=('Helvetica', 12))
            price_label.pack(pady=5)

            # Trend Indicators
            ttk.Label(frame, textvariable=self.trend_vars[symbol], font=('Helvetica', 10)).pack(pady=5)

            # Small Price History Graph
            fig = plt.Figure(figsize=(3, 2), dpi=100)
            ax = fig.add_subplot(111)
            ax.axis('off')
            canvas_plot = FigureCanvasTkAgg(fig, master=frame)
            canvas_plot.draw()
            canvas_plot.get_tk_widget().pack(pady=5)

            # Store the figure for updating
            self.trend_vars[symbol].fig = fig
            self.trend_vars[symbol].ax = ax
            self.trend_vars[symbol].canvas_plot = canvas_plot

        # Market Sentiment Frame
        sentiment_frame = ttk.LabelFrame(self.home_frame, text="Market Sentiment", padding=10)
        sentiment_frame.pack(pady=10, padx=20, fill=tk.X)

        ttk.Label(sentiment_frame, text="Average Sentiment:", font=('Helvetica', 12, 'bold')).pack(anchor='w',
                                                                                                   padx=20, pady=5)
        ttk.Label(sentiment_frame, textvariable=self.sentiment_var_home, font=('Helvetica', 14)).pack(anchor='w',
                                                                                                     padx=20, pady=5)

        # Info Button within Sentiment Frame
        info_button = ttk.Button(sentiment_frame, text="More Info", command=self.show_sentiment_info)
        info_button.pack(anchor='e', padx=20, pady=5)

        # Export Data Button
        export_data_button = ttk.Button(self.home_frame, text="Export Data", command=self.export_live_data)
        export_data_button.pack(pady=10)

        # Export Chart Button
        export_chart_button = ttk.Button(self.home_frame, text="Export Charts", command=self.export_all_charts)
        export_chart_button.pack(pady=10)

    def export_live_data(self):
        """
        Export live price and trend indicators to CSV files.
        """
        try:
            # Gather live price
            data_price = {
                'Cryptocurrency': [],
                'Price (USD)': []
            }
            for symbol in self.cryptocurrency_symbols:
                price = self.price_vars[symbol].get()
                data_price['Cryptocurrency'].append(symbol)
                data_price['Price (USD)'].append(price.replace('$', '').replace(',', '') if price != "N/A" else "N/A")
            df_price = pd.DataFrame(data_price)

            # Gather trend indicators
            data_trend = {
                'Cryptocurrency': [],
                'MA50': [],
                'MA200': []
            }
            for symbol in self.cryptocurrency_symbols:
                trend_info = self.trend_vars[symbol].get()
                parts = trend_info.split('|')
                ma50 = parts[0].split(': ')[1] if len(parts) > 0 else "N/A"
                ma200 = parts[1].split(': ')[1] if len(parts) > 1 else "N/A"
                data_trend['Cryptocurrency'].append(symbol)
                data_trend['MA50'].append(ma50)
                data_trend['MA200'].append(ma200)
            df_trend = pd.DataFrame(data_trend)

            # Export to separate CSV files
            df_price.to_csv('Live_Price.csv', index=False)
            df_trend.to_csv('Trend_Indicators.csv', index=False)

            messagebox.showinfo("Export Successful",
                                f"Data exported to 'Live_Price.csv' and 'Trend_Indicators.csv' successfully!")
            send_notification("Export Successful",
                              f"Data exported to 'Live_Price.csv' and 'Trend_Indicators.csv' successfully!")
        except Exception as e:
            messagebox.showerror("Export Failed", f"An error occurred: {e}")
            send_notification("Export Failed", f"An error occurred: {e}")

    def export_all_charts(self):
        """
        Export all small price history charts on the home page to PNG files.
        """
        try:
            for symbol in self.cryptocurrency_symbols:
                fig = self.trend_vars[symbol].fig
                filename = f"{symbol}_price_history.png"
                fig.savefig(filename)
            messagebox.showinfo("Export Successful", f"All charts exported successfully!")
            send_notification("Export Successful", f"All charts exported successfully!")
        except Exception as e:
            messagebox.showerror("Export Failed", f"An error occurred while exporting charts: {e}")
            send_notification("Export Failed", f"An error occurred while exporting charts: {e}")

    def show_sentiment_info(self):
        """
        Display the latest news articles related to market sentiment.
        """
        # Fetch the latest news articles
        def fetch_and_prepare_news():
            try:
                articles = fetch_market_news(query="cryptocurrency", language="en", page_size=10)
                if not articles:
                    self.queue.put(lambda: messagebox.showinfo("News Info", "No news articles available at the moment."))
                    return

                # Process each article with sentiment categorization
                processed_articles = []
                analyzer = SentimentIntensityAnalyzer()
                for article in articles:
                    title = article.get('title', 'No Title')
                    description = article.get('description', 'No Description')
                    url = article.get('url', '#')

                    content = description if description else title
                    sentiment_score = analyzer.polarity_scores(content)['compound']
                    sentiment_category = categorize_sentiment(sentiment_score)

                    processed_articles.append({
                        'title': title,
                        'description': description,
                        'url': url,
                        'sentiment': sentiment_category
                    })

                # Put the processed articles into the queue for GUI rendering
                self.queue.put(lambda: self.create_sentiment_info_window(processed_articles))
            except Exception as e:
                self.queue.put(lambda e=e: messagebox.showerror("Error", f"Failed to fetch news articles: {e}"))
                send_notification("Error", f"Failed to fetch news articles: {e}")

        # Start the thread to fetch and prepare news data
        Thread(target=fetch_and_prepare_news).start()

    def create_sentiment_info_window(self, articles):
        """
        Create the "More Info" pop-up window with the list of articles and their sentiments.
        """
        try:
            info_window = tk.Toplevel()
            info_window.title("Market Sentiment News")
            info_window.geometry("800x600")

            # Apply current theme to the pop-up
            theme = self.style.theme_use()
            if theme in ['arc', 'plastik', 'clearlooks']:
                background = '#2E2E2E'
                foreground = 'white'
            else:
                background = 'white'
                foreground = 'black'
            info_window.configure(bg=background)

            # Add a scrollable frame
            canvas = tk.Canvas(info_window, bg=background)
            scrollbar = ttk.Scrollbar(info_window, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(
                    scrollregion=canvas.bbox("all")
                )
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            for article in articles:
                # Article Frame
                article_frame = ttk.Frame(scrollable_frame, padding=10)
                article_frame.pack(fill='x', pady=5, padx=10)

                # Background color based on theme
                article_frame.configure(style='TFrame')

                # Title
                ttk.Label(article_frame, text=f"Title: {article['title']}", font=('Helvetica', 12, 'bold')).pack(anchor='w')

                # Description
                ttk.Label(article_frame, text=f"Description: {article['description']}", font=('Helvetica', 12)).pack(anchor='w')

                # Sentiment
                sentiment_label = ttk.Label(article_frame, text=f"Sentiment: {article['sentiment']}", font=('Helvetica', 12, 'bold'))
                if article['sentiment'] == 'Positive':
                    sentiment_label.configure(style='Positive.TLabel')
                elif article['sentiment'] == 'Negative':
                    sentiment_label.configure(style='Negative.TLabel')
                else:
                    sentiment_label.configure(style='Neutral.TLabel')
                sentiment_label.pack(anchor='w')

                # Read more link
                url = article['url']
                link_label = ttk.Label(article_frame, text=f"Read more: {url}", foreground='cyan', cursor="hand2")
                link_label.pack(anchor='w')
                link_label.bind("<Button-1>", lambda e, url=url: webbrowser.open_new(url))

            # Add a close button
            close_button = ttk.Button(info_window, text="Close", command=info_window.destroy)
            close_button.pack(pady=10)
        except Exception as e:
            self.queue.put(lambda e=e: messagebox.showerror("Error", f"Failed to create sentiment info window: {e}"))
            send_notification("Error", f"Failed to create sentiment info window: {e}")

    def setup_predictions_tab(self):
        # Header
        ttk.Label(self.prediction_frame, text="Bitcoin Price Predictions", style='Header.TLabel').pack(pady=20)

        # Prediction Buttons
        button_frame_predictions = ttk.Frame(self.prediction_frame)
        button_frame_predictions.pack(pady=20)

        short_term_button = ttk.Button(button_frame_predictions, text="Show Short-term Prediction",
                                       command=lambda: self.show_prediction(short_term=True))
        short_term_button.grid(row=0, column=0, padx=20, pady=10)

        long_term_button = ttk.Button(button_frame_predictions, text="Show Long-term Prediction",
                                      command=lambda: self.show_prediction(short_term=False))
        long_term_button.grid(row=0, column=1, padx=20, pady=10)

        # Export Predictions Button
        export_predictions_button = ttk.Button(self.prediction_frame, text="Export Predictions",
                                               command=self.export_predictions)
        export_predictions_button.pack(pady=10)

    def export_predictions(self):
        """
        Export prediction data to CSV.
        """
        try:
            # Implement logic to gather prediction data
            if hasattr(self, 'current_prediction_df') and not self.current_prediction_df.empty:
                self.current_prediction_df.to_csv('BTC_Predictions.csv', index=False)
                messagebox.showinfo("Export Successful",
                                    f"Predictions exported to 'BTC_Predictions.csv' successfully!")
                send_notification("Export Successful",
                                  f"Predictions exported to 'BTC_Predictions.csv' successfully!")
            else:
                messagebox.showerror("Export Error", "No prediction data available to export.")
                send_notification("Export Error", "No prediction data available to export.")
        except Exception as e:
            messagebox.showerror("Export Failed", f"An error occurred: {e}")
            send_notification("Export Failed", f"An error occurred: {e}")

    def setup_wallet_tab(self):
        # Header
        ttk.Label(self.wallet_frame, text="Wallet Management", style='Header.TLabel').pack(pady=20)

        # Setup Wallet Button
        setup_wallet_button = ttk.Button(self.wallet_frame, text="Setup Wallet",
                                         command=self.setup_wallet_interface)
        setup_wallet_button.pack(pady=20)

        # Additional wallet-related functionalities can be added here

    def setup_wallet_interface(self):
        """
        Setup wallet interface for user input.
        """
        wallet_window = tk.Toplevel()
        wallet_window.title("Crypto Exchange")
        wallet_window.geometry("500x450")
        wallet_window.resizable(False, False)

        # Apply current theme to the pop-up
        theme = self.style.theme_use()
        if theme in ['arc', 'plastik', 'clearlooks']:
            background = '#2E2E2E'
            foreground = 'white'
        else:
            background = 'white'
            foreground = 'black'
        wallet_window.configure(bg=background)

        # Styling
        style = ttk.Style(wallet_window)
        style.configure('TLabel', font=('Helvetica', 12), background=background, foreground=foreground)
        style.configure('TButton', font=('Helvetica', 12))

        # Header
        ttk.Label(wallet_window, text="Crypto Exchange", style='Header.TLabel').pack(pady=10)

        # Recipient Wallet Address
        ttk.Label(wallet_window, text="Recipient Wallet Address:", background=background, foreground=foreground).pack(
            pady=5, anchor='w', padx=20)
        wallet_entry = ttk.Entry(wallet_window, width=60)
        wallet_entry.pack(pady=5, padx=20)

        # Amount of BTC to send
        ttk.Label(wallet_window, text="Amount of BTC to send:", background=background, foreground=foreground).pack(
            pady=5, anchor='w', padx=20)
        amount_entry = ttk.Entry(wallet_window, width=25)
        amount_entry.pack(pady=5, padx=20)

        # Currency selection
        ttk.Label(wallet_window, text="Currency to exchange to:", background=background, foreground=foreground).pack(
            pady=5, anchor='w', padx=20)
        currency_var = tk.StringVar(wallet_window)
        currency_var.set("USDT")  # Default value
        currency_options = ["USDT", "ETH", "LTC", "XRP", "BNB", "ADA", "SOL"]  # Add more as needed
        currency_dropdown = ttk.OptionMenu(wallet_window, currency_var, currency_options[0], *currency_options)
        currency_dropdown.pack(pady=5, padx=20)

        # Exchange rate display
        exchange_rate_var = tk.StringVar(wallet_window)
        exchange_rate_label = ttk.Label(wallet_window, textvariable=exchange_rate_var,
                                        font=("Helvetica", 12, "bold"), background=background,
                                        foreground=foreground)
        exchange_rate_label.pack(pady=10)

        def update_exchange_rate():
            from_symbol = "BTC"
            to_symbol = currency_var.get().upper()
            try:
                rate_info = get_exchange_rate(from_symbol, to_symbol)
                if rate_info and 'rate' in rate_info:
                    rate = rate_info['rate']
                    exchange_rate_var.set(f"Exchange Rate: 1 BTC = {rate:.2f} {to_symbol}")
                else:
                    exchange_rate_var.set("Failed to fetch exchange rate.")
            except Exception as e:
                exchange_rate_var.set("Error fetching rate.")

        update_exchange_rate()

        def on_currency_change(*args):
            update_exchange_rate()

        currency_var.trace('w', on_currency_change)

        # Submit button
        def on_submit():
            wallet_address = wallet_entry.get().strip()
            btc_amount = amount_entry.get().strip()
            to_symbol = currency_var.get().strip().upper()

            if not wallet_address:
                messagebox.showerror("Input Error", "Please enter the recipient wallet address.")
                return
            if not btc_amount:
                messagebox.showerror("Input Error", "Please enter the amount of BTC to send.")
                return
            try:
                btc_amount_float = float(btc_amount)
                if btc_amount_float <= 0:
                    messagebox.showerror("Input Error", "Amount must be positive.")
                    return
            except ValueError:
                messagebox.showerror("Input Error", "Please enter a valid number for the amount.")
                return

            execute_trade(wallet_address, btc_amount_float, to_symbol)
            messagebox.showinfo("Trade Executed",
                                f"Sent {btc_amount_float} BTC to {wallet_address} as {to_symbol}.")
            send_notification("Trade Executed",
                              f"Sent {btc_amount_float} BTC to {wallet_address} as {to_symbol}.")

        submit_button = ttk.Button(wallet_window, text="Submit", command=on_submit)
        submit_button.pack(pady=20)

    def setup_info_tab(self):
        ttk.Label(self.info_frame, text="Information", style='Header.TLabel').pack(pady=20)

        info_text = (
            "📚 **Crypto Dashboard Information**\n\n"
            "🔍 **Price Sentiment:**\n"
            "Price sentiment reflects the overall market mood towards a cryptocurrency. Positive sentiment can drive prices up as more investors are optimistic. Negative sentiment can lead to price drops due to pessimism.\n\n"
            "📈 **Trend Indicators:**\n"
            "- **MA50:** 50-day Moving Average. It smoothens out price data by calculating the average price over the last 50 days.\n"
            "- **MA200:** 200-day Moving Average. Similar to MA50 but over a longer period, providing insights into long-term trends.\n\n"
            "🔵 **Bollinger Bands:**\n"
            "These are volatility bands placed above and below a moving average. They expand and contract based on market volatility.\n\n"
            "🛠️ **How Values are Calculated:**\n"
            "All trend indicators are derived from historical price data. Sentiment analysis is performed on recent news headlines to gauge market mood."
        )

        # Apply current theme to the info tab
        theme = self.style.theme_use()
        if theme in ['arc', 'plastik', 'clearlooks']:
            background = '#2E2E2E'
            foreground = 'white'
        else:
            background = 'white'
            foreground = 'black'

        text_widget = tk.Text(self.info_frame, wrap='word', font=('Helvetica', 12), bg=background, fg=foreground)
        text_widget.insert(tk.END, info_text)
        text_widget.config(state='disabled')  # Make the text read-only
        text_widget.pack(expand=True, fill='both', padx=20, pady=10)

    def setup_settings_tab(self):
        ttk.Label(self.settings_frame, text="Settings", style='Header.TLabel').pack(pady=20)

        # Theme Selection
        theme_selection_frame = ttk.LabelFrame(self.settings_frame, text="Theme Selection", padding=10)
        theme_selection_frame.pack(pady=10, padx=20, fill=tk.X)

        themes = self.style.theme_names()
        theme_var = tk.StringVar(value=self.style.theme_use())

        for theme in themes:
            ttk.Radiobutton(theme_selection_frame, text=theme.capitalize(), variable=theme_var, value=theme,
                           command=lambda t=theme: self.switch_theme(t)).pack(anchor='w', pady=2)

        # Notification Preferences 
        notification_frame = ttk.LabelFrame(self.settings_frame, text="Notification Preferences", padding=10)
        notification_frame.pack(pady=10, padx=20, fill=tk.X)

        self.notifications_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(notification_frame, text="Enable Notifications", variable=self.notifications_var).pack(anchor='w',
                                                                                                               pady=5,
                                                                                                               padx=20)

    def switch_theme(self, selected_theme):
        """
        Switch the theme of the application.
        """
        self.style.set_theme(selected_theme)
        self.status_var.set(f"Theme switched to {selected_theme.capitalize()}")
        send_notification("Theme Changed", f"Theme switched to {selected_theme.capitalize()}")

    def login(self):
        """
        Display the login window for user authentication.
        """
        auth_window = tk.Toplevel()
        auth_window.title("Login")
        auth_window.geometry("400x300")
        auth_window.resizable(False, False)

        # Center the login window
        auth_window.update_idletasks()
        width = 400
        height = 300
        x = (auth_window.winfo_screenwidth() // 2) - (width // 2)
        y = (auth_window.winfo_screenheight() // 2) - (height // 2)
        auth_window.geometry(f"{width}x{height}+{x}+{y}")

        # Apply current theme to the pop-up
        theme = self.style.theme_use()
        if theme in ['arc', 'plastik', 'clearlooks']:
            background = '#2E2E2E'
            foreground = 'white'
        else:
            background = 'white'
            foreground = 'black'
        auth_window.configure(bg=background)

        # Styling stuff
        style = ttk.Style(auth_window)
        style.configure('TLabel', font=('Helvetica', 12), background=background, foreground=foreground)
        style.configure('TButton', font=('Helvetica', 12))

        # Header stuff
        ttk.Label(auth_window, text="User Login", style='Header.TLabel').pack(pady=20)

        # Username stuff
        ttk.Label(auth_window, text="Username:", background=background, foreground=foreground).pack(
            pady=5, anchor='w', padx=50)
        username_entry = ttk.Entry(auth_window, width=30)
        username_entry.pack(pady=5, padx=50)
        username_entry.focus()

        # Password stuff
        ttk.Label(auth_window, text="Password:", background=background, foreground=foreground).pack(
            pady=5, anchor='w', padx=50)
        password_entry = ttk.Entry(auth_window, width=30, show="*")
        password_entry.pack(pady=5, padx=50)

        # Login Button
        def verify_credentials():
            username = username_entry.get()
            password = password_entry.get()

            # Authentication logic
            if username.lower() == "adam" and password == "adam": # JUST PREDEFIEND HERE
                messagebox.showinfo("Login Successful", f"Welcome, {username.capitalize()}!")
                self.greeting_var.set(f"Hello {username.capitalize()}!")
                auth_window.destroy()
            else:
                messagebox.showerror("Login Failed", "Invalid username or password.")

        login_button = ttk.Button(auth_window, text="Login", command=verify_credentials)
        login_button.pack(pady=20)

        # Handle window close event
        def on_close():
            """
            Handle the event when the user tries to close the login window.
            """
            if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
                self.root.destroy()

        auth_window.protocol("WM_DELETE_WINDOW", on_close)

        # Hide the main window while login is active
        self.root.withdraw()

        # Keep the focus on the login window
        auth_window.grab_set()
        self.root.wait_window(auth_window)  # Wait until the window is closed

        # Show the main window after successful login
        self.root.deiconify()

    # --- Real-time Updates --

    def update_live_prices(self):
        """
        Update live cryptocurrency prices periodically.
        """
        def fetch_prices():
            for symbol in self.cryptocurrency_symbols:
                price = self.get_crypto_price(symbol)
                if price is not None:
                    # Schedule the update in the main thread via the queue
                    self.queue.put(lambda s=symbol, p=price: self.price_vars[s].set(f"${p:,.2f}"))
                else:
                    self.queue.put(lambda s=symbol: self.price_vars[s].set("N/A"))
            self.queue.put(lambda: self.status_var.set("Live prices updated."))
            self.queue.put(lambda: send_notification("Live Prices Updated", "Your live cryptocurrency prices have been updated."))

        Thread(target=fetch_prices).start()
        self.root.after(60000, self.update_live_prices)  # Update every 60 seconds

    def get_crypto_price(self, symbol):
        """
        Fetch real-time cryptocurrency price from CoinMarketCap.
        """
        url = CMC_URL
        headers = {
            'X-CMC_PRO_API_KEY': CMC_API_KEY,
            'Accepts': 'application/json'
        }
        try:
            params = {
                'symbol': symbol,
                'convert': 'USD'
            }
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            price = data['data'][symbol]['quote']['USD']['price']
            return round(price, 2)
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred while fetching {symbol} price: {http_err}")
        except Exception as e:
            print(f"Error fetching {symbol} price: {e}")
        return None

    def update_trend_indicators(self):
        """
        Update trend indicators periodically.
        """
        def update_trends():
            for symbol in self.cryptocurrency_symbols:
                try:
                    yfinance_symbol = f"{symbol}-USD"  # Ensure correct symbol format for yfinance
                    data = fetch_and_prepare_historical_data(symbol=yfinance_symbol, interval='1d')
                    if data.empty:
                        self.queue.put(lambda s=symbol: self.trend_vars[s].set("MA50: N/A | MA200: N/A"))
                        continue

                    data = calculate_trend_indicators(data)
                    latest_ma50 = data['MA50'].iloc[-1]
                    latest_ma200 = data['MA200'].iloc[-1]

                    trend_info = f"MA50: {latest_ma50:,.2f} | MA200: {latest_ma200:,.2f}"
                    self.queue.put(lambda s=symbol, info=trend_info: self.trend_vars[s].set(info))

                    # Update small price history graph
                    fig = self.trend_vars[symbol].fig
                    ax = self.trend_vars[symbol].ax
                    ax.clear()
                    ax.plot(data['Date'], data['Close'], label='Close Price', color='#00ff00')
                    ax.plot(data['Date'], data['MA50'], label='MA50', color='#FFD700')
                    ax.plot(data['Date'], data['MA200'], label='MA200', color='#FF4500')
                    ax.legend(loc='upper left', fontsize=8)
                    ax.axis('off')
                    fig.tight_layout()

                    self.queue.put(lambda s=symbol: self.trend_vars[s].canvas_plot.draw())

                except Exception as e:
                    print(f"Error updating trend indicators for {symbol}: {e}")
                    # Properly capture 'e' in the lambda
                    self.queue.put(lambda s=symbol, e=e: self.trend_vars[s].set(f"MA50: N/A | MA200: N/A"))
            self.queue.put(lambda: self.status_var.set("Trend indicators updated."))
            self.queue.put(lambda: send_notification("Trend Indicators Updated", "Trend indicators have been refreshed."))

        Thread(target=update_trends).start()
        self.root.after(120000, self.update_trend_indicators)  # Update every 2 minutes

    def update_market_sentiment(self):
        """
        Update market sentiment periodically.
        """
        def update_sentiment():
            try:
                from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                to_date = datetime.now().strftime('%Y-%m-%d')
                sentiment_percentage, _ = fetch_news_sentiment(query="cryptocurrency", from_date=from_date,
                                                         to_date=to_date)
                self.queue.put(lambda: self.sentiment_var_home.set(f"{sentiment_percentage:.2f}%"))
                self.queue.put(lambda: self.status_var.set("Market sentiment updated."))
                self.queue.put(lambda: send_notification("Sentiment Updated", "Market sentiment has been updated."))
            except Exception as e:
                self.queue.put(lambda: self.sentiment_var_home.set("N/A"))
                self.queue.put(lambda: self.status_var.set("Failed to update market sentiment."))
                self.queue.put(lambda e=e: send_notification("Sentiment Update Failed", f"An error occurred: {e}"))

        Thread(target=update_sentiment).start()
        self.root.after(180000, self.update_market_sentiment)  # Update every 3 minutes

    def process_queue(self):
        """
        Process GUI updates from the queue.
        """
        try:
            while True:
                task = self.queue.get_nowait()
                task()
                self.queue.task_done()
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)  # Check the queue every 100 ms

    # Prediction Functions 

    def show_prediction(self, short_term=True):
        """
        Show prediction for Bitcoin.
        Short_term=True: Includes sentiment data
        Short_term=False: Based on historical price data only
        """
        def run_prediction():
            try:
                self.queue.put(lambda: self.status_var.set("Fetching historical price data..."))
                send_notification("Prediction Started", "Fetching data for prediction...")

                yfinance_symbol = "BTC-USD"
                data = fetch_and_prepare_historical_data(symbol=yfinance_symbol, interval='1d')
                if data.empty:
                    self.queue.put(lambda: messagebox.showerror("Data Error",
                                                                 "No historical data fetched. Please check the symbol and internet connection."))
                    self.queue.put(lambda: self.status_var.set("Data fetching failed."))
                    send_notification("Prediction Failed", "No historical data fetched.")
                    return

                if short_term:
                    # Fetch sentiment data
                    from_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # Past year
                    to_date = datetime.now().strftime('%Y-%m-%d')
                    sentiment_percentage, articles = fetch_news_sentiment(query="Bitcoin", from_date=from_date,
                                                                     to_date=to_date)

                    # Create a sentiment DataFrame
                    sentiment_df = pd.DataFrame(columns=['Date', 'Sentiment'])
                    date_sentiment = {}

                    analyzer = SentimentIntensityAnalyzer()
                    for article in articles:
                        published_at = article['publishedAt'][:10]  # Extract date in 'YYYY-MM-DD' format
                        sentiment = analyzer.polarity_scores(article['description'] or article['title'])['compound']
                        if published_at in date_sentiment:
                            date_sentiment[published_at].append(sentiment)
                        else:
                            date_sentiment[published_at] = [sentiment]

                    sentiment_list = []
                    for date, sentiments in date_sentiment.items():
                        avg_sentiment = sum(sentiments) / len(sentiments)
                        sentiment_percentage = average_sentiment_percentage(avg_sentiment)
                        sentiment_list.append({'Date': pd.to_datetime(date), 'Sentiment': sentiment_percentage})

                    if sentiment_list:
                        sentiment_df_new = pd.DataFrame(sentiment_list)
                        if not sentiment_df_new.empty:
                            sentiment_df = pd.concat([sentiment_df, sentiment_df_new], ignore_index=True)
                    else:
                        sentiment_df = pd.DataFrame(columns=['Date', 'Sentiment'])

                    # Merge sentiment with price data
                    data = data.copy()  # To avoid SettingWithCopyWarning
                    data['Date'] = pd.to_datetime(data['Date'])  # Ensure 'Date' is datetime
                    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
                    data = pd.merge(data, sentiment_df, on='Date', how='left')
                    data['Sentiment'] = data['Sentiment'].fillna(data['Sentiment'].mean())  # Fill NaN with average sentiment

                self.queue.put(lambda: self.status_var.set("Training the prediction model..."))
                send_notification("Training Model", "Training the prediction model.")

                forecast_horizon = 30 if short_term else 90  # 30 days short-term, 90 days long-term
                include_sentiment = short_term

                model, train_data, test_data, future_data = build_and_train_lstm(
                    data, forecast_horizon=forecast_horizon, include_sentiment=include_sentiment
                )

                title = "Short-term Bitcoin Price Prediction" if short_term else "Long-term Bitcoin Price Prediction"

                self.queue.put(lambda: self.status_var.set(f"Displaying {title}"))
                send_notification("Prediction Ready", f"{title} is ready.")

                # Store current prediction data for export
                self.current_prediction_df = pd.DataFrame({
                    'Date': pd.concat([train_data['Date'], test_data['Date'], future_data['Date']]),
                    'Actual Close': pd.concat(
                        [train_data['Close'], test_data['Close'], pd.Series([np.nan] * forecast_horizon)]),
                    'Predicted Close': pd.concat(
                        [train_data.get('Predicted_Close', pd.Series([np.nan] * len(train_data))),
                         test_data['Predicted_Close'],
                         future_data['Predicted_Close']])
                }).reset_index(drop=True)

                self.queue.put(lambda t=title, td=train_data, te=test_data, tf=future_data: self.plot_predictions(t, td, te, tf))
            except Exception as e:
                self.queue.put(lambda e=e: messagebox.showerror("Prediction Error", f"An error occurred: {e}"))
                self.queue.put(lambda: self.status_var.set("Prediction failed."))
                self.queue.put(lambda e=e: send_notification("Prediction Failed", f"An error occurred: {e}"))

        Thread(target=run_prediction).start()

    def plot_predictions(self, title, train_data, test_data, future_data):
        """
        Plot predictions for train, test, and future data.
        """
        try:
            plot_window = tk.Toplevel()
            plot_window.title(title)
            plot_window.geometry("1000x600")
            fig = plt.Figure(figsize=(10, 5), dpi=100)
            ax = fig.add_subplot(111)

            ax.plot(train_data['Date'], train_data['Close'], label='Actual Prices (Train)', color='#00ff00')
            if 'Predicted_Close' in train_data.columns and not train_data['Predicted_Close'].isna().all():
                ax.plot(train_data['Date'], train_data['Predicted_Close'], label='Predicted Prices (Train)', color='#00CED1')

            ax.plot(test_data['Date'], test_data['Close'], label='Actual Prices (Test)', color='#ff7f50')
            ax.plot(test_data['Date'], test_data['Predicted_Close'], label='Predicted Prices (Test)', color='#1E90FF')

            # Plot future predictions if available
            if not future_data['Predicted_Close'].isna().all():
                ax.plot(future_data['Date'], future_data['Predicted_Close'],
                        label='Predicted Future Prices', color='#9400D3')

            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price (USD)', fontsize=12)
            ax.legend()
            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Store the current figure for export
            self.current_fig = fig

            # Export Chart Button within Plot Window
            export_chart_button = ttk.Button(plot_window, text="Export Chart",
                                            command=lambda: export_chart(fig, f"{title.replace(' ', '_')}.png"))
            export_chart_button.pack(pady=10)

            # Add a close button with modern styling
            close_button = ttk.Button(plot_window, text="Close", command=plot_window.destroy)
            close_button.pack(pady=10)
        except KeyError as e:
            # Properly capture 'e' in the lambda
            self.queue.put(lambda e=e: messagebox.showerror("Plot Error", f"Missing data for plotting: {e}"))
            send_notification("Plot Error", f"Missing data for plotting: {e}")
        except Exception as e:
            # Properly capture 'e' in the lambda
            self.queue.put(lambda e=e: messagebox.showerror("Plot Error", f"An error occurred while plotting predictions: {e}"))
            self.queue.put(lambda e=e: send_notification("Plot Error", f"An error occurred while plotting predictions: {e}"))

    # - Additional Features ---

    def send_notification_feature(self, title, message):
        """
        Send desktop notification.
        """
        send_notification(title, message)


# --- Helper Utility Functions ----


def average_sentiment_percentage(sentiment_score):
    """
    Convert sentiment score from [-1,1] to [0,100] percentage.

    Args:
        sentiment_score (float): Average sentiment score.

    Returns:
        float: Sentiment percentage.
    """
    return ((sentiment_score + 1) / 2) * 100


# ================== Main Function ==================


def main():
    """
    Main function to initialize data and launch the GUI.
    """
    # Ensure API keys are defined
    if CMC_API_KEY == "YOUR_COINMARKETCAP_API_KEY" or NEWSAPI_KEY == "YOUR_NEWSAPI_KEY":
        print("Error: Please replace the placeholder API keys with your actual API keys.")
        return

    root = tk.Tk()

  
    app = CryptoDashboard(root)

    app.process_queue()

    app.login()

    root.mainloop()


if __name__ == "__main__":
    main()
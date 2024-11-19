# BitcoinML-Buy-Sell

A crypto wallet app with news prediction, including short-term and long-term predictions, and executions.

## Final Notes

- **Model Accuracy**:  
  The accuracy of the model is not fully optimized and is unlikely to exceed 75% without sacrificing the usability of the application.
  
  - The short-term model must be retrained each time it is run because it relies on news data from the current day. To avoid long wait times for predictions, it is trained with a low number of epochs.
  - The long-term model is based primarily on price trends but also incorporates news data.
  
  - **Short-Term Model Accuracy**:  
    At one point, the short-term model showed high accuracy when trained with a larger number of epochs, which was a moment of success. However, this implementation cannot be used in the real app because it takes hours or even days to train.

## Code Explanation

### `Main.py`
- The main driver of the application responsible for:
  - Fetching cryptocurrency data.
  - Analyzing it and predicting future prices.
  - Connecting to APIs to retrieve the latest prices and historical data for BTC (Bitcoin), as well as displaying information for the 4 other main cryptocurrencies (USDT, ETH, SOL, BNB).
  - Analyzing news articles related to crypto to gauge market sentiment (positive or negative).
  - Using an ML model trained on historical data and current news sentiment to predict future prices.
  - Providing a simple UI to display current prices, market sentiment, predictions, wallet setup, and trading functionality.
    - Includes themes, a settings page, an info page, and user login.

### `Wallet.py`
- Manages the crypto wallet, handling transactions and balances.
- Facilitates exchanges using the **StealthFX API**.
- Supports executing buy/sell orders in a unique way, as described earlier in the document.

## Future Hopes

- **Expansion**:
  - Expand the app to include stock market predictions and additional crypto predictions.
  
- **Improved Accuracy**:
  - Explore using APIs for faster training with high-end GPU infrastructure to train with a high epoch rate every time the user clicks "short-term" predictions, improving accuracy while maintaining the model’s reliance on news data.

- **Cross-Platform Expansion**:
  - Create mobile apps for iOS and Android, as well as a MacOS app, making the application accessible to a broader audience.

## Video Demo

[Watch the demo video](https://eastsidepreparatory-my.sharepoint.com/:v:/g/personal/adalloul_eastsideprep_org/EW5s21ts3PxNsHOU2j4qyAkBxEsJXuY3gIHbvHOL5EYmpQ?e=JuOO6H)

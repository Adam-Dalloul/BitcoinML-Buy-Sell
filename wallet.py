import requests
import json

# Coinmarket stuff
API_KEY = "-" # REPLACE WITH ACTUAL ONE BECAUSE THIS IS PUBLIC REPO
BASE_URL = "https://pro-api.coinmarketcap.com"

# SealthFX API
SEALTHFX_API = "-" # REPLACE WITH ACTUAL ONE BECAUSE THIS IS PUBLIC REPO
SEALTHFX_BASE_URL = "https://api.sealthfx.com"

def get_exchange_rate(from_symbol, to_symbol):
    """Get exchange rate from one cryptocurrency to another using CoinMarketCap API."""
    url = f"{BASE_URL}/v1/cryptocurrency/quotes/latest"
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': API_KEY,
    }
    parameters = {
        'symbol': from_symbol,
        'convert': to_symbol
    }

    try:
        response = requests.get(url, headers=headers, params=parameters)
        data = response.json()

        # Check if the response contains the expected data structure
        if 'data' in data and from_symbol in data['data']:
            quote = data['data'][from_symbol]['quote']
            if to_symbol in quote:
                price = quote[to_symbol]['price']
                return {'rate': price}
            else:
                print(f"Conversion to {to_symbol} not available.")
                return None
        else:
            print(f"Error fetching exchange rate: {data}")
            return None
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None

def execute_trade(from_currency, to_currency, amount, wallet_address):
    """executing a trade."""
    print(f"Executing trade: {amount} {from_currency} to {to_currency} for wallet {wallet_address}")
    # Return trade info
    return {
        "status": "success",
        "from_currency": from_currency,
        "to_currency": to_currency,
        "amount": amount,
        "wallet_address": wallet_address
    }

def display_exchange_rates():
    """Display exchange rates for various cryptocurrencies and amounts."""
    cryptos = ["BTC", "ETH", "USDC"]
    target_currency = "USDT"
    amounts = [1, 0.5, 0.25]

    for crypto in cryptos:
        rate = get_exchange_rate(crypto, target_currency)
        if rate is not None:
            print(f"\nExchange rates for {crypto} to {target_currency}:")
            for amount in amounts:
                print(f"  {amount} {crypto} = {amount * rate['rate']:.2f} {target_currency}")
        else:
            print(f"Failed to retrieve the exchange rate for {crypto}.")

def exchange_currency_sealthfx(from_currency, to_currency, amount):
    """Exchange currency using SealthFX API."""
    url = f"{SEALTHFX_BASE_URL}/v1/exchange"
    headers = {
        'Authorization': f'Bearer {SEALTHFX_API}',
        'Content-Type': 'application/json'
    }
    payload = {
        'from_currency': from_currency,
        'to_currency': to_currency,
        'amount': amount
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()

        if response.status_code == 200:
            print(f"Exchange successful: {amount} {from_currency} to {to_currency}")
            return data
        else:
            print(f"Exchange failed: {data.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"Exception occurred during exchange: {e}")
        return None

def main():
    display_exchange_rates()
    exchange_result = exchange_currency_sealthfx("BTC", "ETH", 0.1)
    if exchange_result:
        print("Exchange result:", exchange_result)

# run
if __name__ == "__main__":
    main()

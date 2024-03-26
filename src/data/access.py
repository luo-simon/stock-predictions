"""Ingests the data without dealing with missing values, or outliers."""

import argparse
import os.path
import yfinance as yf
from datetime import datetime

def download_raw_data(ticker, path):
    """Calls yfinance API to retrieve desired data and stores as .csv file"""
    stock = yf.Ticker(ticker)
    df = stock.history(period="max")
    df.to_csv(os.path.join(path, f"{ticker.upper()}.csv"), index=True, header=True)

def data(ticker="aapl", start=datetime(2018,1,1), end=datetime(2023,1,1)):
    """Get stock data within a specified date range"""
    print(ticker, start, end)
    stock = yf.Ticker(ticker)
    return stock.history(start=start, end=end)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("-t", "--ticker", help="Ticker symbol", required=True)
    parser.add_argument("-r", "--raw-path", help="Raw data path", required=True)
    args = parser.parse_args()
    download_raw_data(ticker=args.ticker, path=args.raw_path)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("-t", "--ticker", help="Ticker symbol", required=True)
    parser.add_argument("-r", "--raw-path", help="Raw data path", default="data/raw")
    args = parser.parse_args()
    download_raw_data(ticker=args.ticker, path=args.raw_path)
"""Ingests the data without dealing with missing values, or outliers."""

import argparse
import os.path
import yfinance as yf
from datetime import datetime

def download_raw_data(ticker="aapl", start=datetime(2018,1,1), end=datetime(2023,1,1), path=''):
    """Read the data from the web or local file, returning structured format such as a data frame"""
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)
    df.to_csv(os.path.join(path, 'train.csv'), index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("-t", "--ticker", help="Ticker symbol", required=True)
    parser.add_argument("-r", "--raw-path", help="Raw data path", required=True)
    args = parser.parse_args()
    download_raw_data(ticker=args.ticker, path=args.raw_path)
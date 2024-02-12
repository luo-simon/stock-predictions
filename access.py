"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

import yfinance as yf
from datetime import datetime

def data(ticker="aapl", start=datetime(2018,1,1), end=datetime(2023,1,1)):
    """Read the data from the web or local file, returning structured format such as a data frame"""
    stock = yf.Ticker(ticker)
    return stock.history(start=start, end=end)
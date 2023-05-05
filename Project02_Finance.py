import pandas_datareader as pdr
import pandas as pd
import numpy as np
import datetime

start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016, 1, 1)

# Bank of America
BAC = data.DataReader("BAC", 'google', start, end)
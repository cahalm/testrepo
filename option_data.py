import numpy as np
import pandas as pd
from datetime import datetime

# sp500_path = '.\Resources\Stock and Index Options\SP500 Option.csv'
sp500_path = '/Users/cahalmurphy/Documents/UCD Files/Stage 4/Semester_2/FIN30190 - Financial Economics II/Project_B/Resources/Stock and Index Options/SP500 Option.csv'

data_path = './Data/data.csv'

# equity_path = '.\Resources\Stock and Index Options\SA9.csv'

headers = ['Date', 'SecurityID', 'SecurityPrice', 'TotalReturn', 'AdjustmentFactor', 'AdjustmentFactor2',
           'InterestRate' ,'Expiration' ,'Strike' ,'OptionID', 'CallPut', 'BestBid', 'BestOffer', 'ImpliedVolatility',
           'Delta', 'Gamma', 'Vega', 'Theta']

dtypes = {'Date': str,
          'SecurityID': int,
          'SecurityPrice': np.float32,
          'TotalReturn': np.float32,
          'AdjustmentFactor': int,
          'AdjustmentFactor2': np.float32,
          'InterestRate': np.float32,
          'Expiration': str,
          'Strike': np.float32,
          'OptionID': int,
          'CallPut': str,
          'BestBid': np.float32,
          'BestOffer': np.float32,
          'ImpliedVolatility': np.float32,
          'Delta': np.float32,
          'Gamma': np.float32,
          'Vega': np.float32,
          'Theta': np.float32,
          }




def _dateparser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def get_data(path: str = sp500_path):
    sp500_data = pd.read_csv(filepath_or_buffer=path, header=None, names=headers,
                             dtype=dtypes,
                             na_values=np.float32(-99.99),
                             parse_dates=['Date', 'Expiration'], date_parser=_dateparser,
                             )

    data = sp500_data.copy()
    # NA Values from CSV file
    nan_number = np.float32(-99.99)
    data = data.replace(nan_number, np.NaN)

    data['Strike'] = data['Strike'] / 1000
    data = data.set_index(['Date', 'CallPut', 'Expiration', 'Strike'], drop=False)
    data = data.sort_index()

    # Columns Editing
    data["tau"] = (data["Expiration"] - data["Date"]).dt.days / 252
    data["mid"] = (data["BestBid"] + data["BestOffer"]) / 2
    data["InterestRate"] /= 100

    return data


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def get_ATM_option(options, delta_value=0.5):
    """
    # Make sure to set Delta to -0.5 for PUTS *********
    """
    if options.CallPut.iloc[0] == 'C' and delta_value < 0:
        print("Negative Delta_Value for finding ATM CALLS")
    if options.CallPut.iloc[0] == 'P' and delta_value > 0:
        print("Positive Delta_Value for finding ATM PUTS")

    delta, index = find_nearest(options.Delta.values, delta_value)

    return options.index[index]  # will be the strike of nearest Delta to 0.5


def get_test_df():
    test_strikes = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    test_vols = [45.6, 41.6, 37.9, 36.6, 37.8, 39.2, 40.0]
    test_df = pd.DataFrame(index=test_strikes, columns=["Strike", "IV"],
                           data={'Strike': test_strikes, 'ImpliedVolatility': test_vols})
    test_df /= 100
    test_df.index /= 100
    return test_df

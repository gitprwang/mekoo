"""Preprocessing Cryptos dataset."""
import numpy as np
import pandas as pd

# Manually download "train.csv" and "asset_details.csv"
# from https://www.kaggle.com/competitions/g-research-crypto-forecasting/data
# to current 'Cryptos' folder.


def fill_empty(frame):
  """removes empty rows.

  Args:
    frame: panda frame

  Returns:
    return a panda frame with empty rows removed

  """
  return frame.reindex(
      range(frame.index[0], frame.index[-1] + 60, 60), method="pad")


crypto_df = pd.read_csv("train.csv")
asset_detail = pd.read_csv("asset_details.csv")

cryptos_data_train = []
cryptos_data_test = []

# there are 14 stocks in total
for asset_id in range(14):
  crypto_df[crypto_df["Asset_ID"] == asset_id].set_index("timestamp")
  df = crypto_df[crypto_df["Asset_ID"] == asset_id].set_index("timestamp")

  # The target is 15-min residulized future returns
  # We need to shift this feature up by 15 rows
  # so that each data entry doesn't contain future information.
  df["Target"] = df["Target"].shift(15)
  df = df.fillna(0)
  df = df.values[20:-1, 1:]

  # Remove inf values
  df[df == float("Inf")] = 0
  df[df == float("-Inf")] = 0

  # split train/test sets
  cryptos_data_train.append(df[:-10000])
  cryptos_data_test.append(df[-10000:])

np.save("train.npy", cryptos_data_train)
np.save("test.npy", cryptos_data_test)

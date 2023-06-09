"""
@Description: Acquiring data and generating features
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-31 11:30:10
"""
import pandas as pd
from pandas import DataFrame
my_data = pd.read_csv('../data/20051201_20051210.csv', index_col='Date')
# print(my_data)


def change_str_to_float(df: DataFrame) -> DataFrame:
    # df.sort_index(ascending=False)
    df['Open'] = df['Open'].str.replace(',', '').astype('float')
    df['Close'] = df['Close'].str.replace(',', '').astype('float')
    df['High'] = df['High'].str.replace(',', '').astype('float')
    df['Low'] = df['Low'].str.replace(',', '').astype('float')
    df['Volume'] = df['Volume'].str.replace('M', 'e6').astype('float')


# change_str_to_float(my_data)


def add_original_feature(df, df_new):
    df_new['open'] = df['Open']
    df_new['open_1'] = df['Open'].shift(1)
    df_new['close_1'] = df['Close'].shift(1)
    df_new['high_1'] = df['High'].shift(1)
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)


def add_avg_Close(df, df_new):
    df_new['avg_close_5'] = df['Close'].rolling(5).mean().shift(1)
    df_new['avg_close_30'] = df['Close'].rolling(21).mean().shift(1)
    df_new['avg_close_365'] = df['Close'].rolling(252).mean().shift(1)
    df_new['ratio_avg_close_5_30'] = df_new['avg_close_5'] / \
        df_new['avg_close_30']
    df_new['ratio_avg_close_5_365'] = df_new['avg_close_5'] / \
        df_new['avg_close_365']
    df_new['ratio_avg_close_30_365'] = df_new['avg_close_30'] / \
        df_new['avg_close_365']


def add_avg_volume(df, df_new):
    df_new['avg_volume_5'] = df['Volume'].rolling(5).mean().shift(1)
    df_new['avg_volume_30'] = df['Volume'].rolling(21).mean().shift(1)
    df_new['avg_volume_365'] = df['Volume'].rolling(252).mean().shift(1)
    df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / \
        df_new['avg_volume_30']
    df_new['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / \
        df_new['avg_volume_365']
    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / \
        df_new['avg_volume_365']


def add_std_Close(df, df_new):
    df_new['std_close_5'] = df['Close'].rolling(5).std().shift(1)
    df_new['std_close_30'] = df['Close'].rolling(21).std().shift(1)
    df_new['std_close_365'] = df['Close'].rolling(252).std().shift(1)
    df_new['ratio_std_close_5_30'] = df_new['std_close_5'] / \
        df_new['std_close_30']
    df_new['ratio_std_close_5_365'] = df_new['std_close_5'] / \
        df_new['std_close_365']
    df_new['ratio_std_close_30_365'] = df_new['std_close_30'] / \
        df_new['std_close_365']


def add_std_volume(df, df_new):
    df_new['std_volume_5'] = df['Volume'].rolling(5).std().shift(1)
    df_new['std_volume_30'] = df['Volume'].rolling(21).std().shift(1)
    df_new['std_volume_365'] = df['Volume'].rolling(252).std().shift(1)
    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / \
        df_new['std_volume_30']
    df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / \
        df_new['std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / \
        df_new['std_volume_365']


def add_return_feature(df, df_new):
    df_new['return_1'] = ((
        (df['Close'] - df['Close'].shift(1))) / df['Close'].shift(1)).shift(1)
    df_new['return_5'] = ((
        (df['Close'] - df['Close'].shift(5))) / df['Close'].shift(5)).shift(1)
    df_new['return_30'] = ((
        (df['Close'] - df['Close'].shift(21))) / df['Close'].shift(21)).shift(1)
    df_new['return_365'] = ((
        (df['Close'] - df['Close'].shift(252))) / df['Close'].shift(252)).shift(1)
    df_new['moving_avg_5'] = df_new['return_1'].rolling(5).mean().shift(1)
    df_new['moving_avg_30'] = df_new['return_1'].rolling(21).mean().shift(1)
    df_new['moving_avg_365'] = df_new['return_1'].rolling(252).mean().shift(1)


def generate_features(df):
    df_new = pd.DataFrame()
    # 6 original features
    add_original_feature(df, df_new)
    # 31 generated features
    add_avg_Close(df, df_new)
    add_avg_volume(df, df_new)
    add_std_Close(df, df_new)
    add_std_volume(df, df_new)
    add_return_feature(df, df_new)
    # the target
    df_new['Close'] = df['Close']
    df_new = df_new.dropna(axis=0)
    return df_new


if __name__ == '__main__':
    data_raw = pd.read_csv('../data/19880101_20191231.csv', index_col='Date')
    change_str_to_float(data_raw)
    data = generate_features(data_raw)
    print(data.round(decimals=3).head(5))

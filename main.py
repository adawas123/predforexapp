# 必要なライブラリをインポート
import numpy as np
import datetime as dt
import streamlit as st
import pandas as pd
import mplfinance as mpf
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
yf.pdr_override()
st.set_option('deprecation.showPyplotGlobalUse', False)


# タイトルとテキストを記入
st.sidebar.header('Forex Prediction App')

#Sidebar：対象選択
st.sidebar.write('1時間データのローソク足を表示します。')
st.sidebar.write('表示対象を選択してください')
brandCode = st.sidebar.selectbox('補足 JPY=X:ドル円、EUR=X:ユーロドル',
    ('JPY=X', 'EUR=X'))

#Sidebar：データ取得ボタン
show_data = st.sidebar.button('ローソク足表示')

#Sidebar：予測開始ボタン
st.sidebar.write('翌日の日本時間17時の価格を予測します。')
get_data = st.sidebar.button('USD-JPYを予測する')
get_data2 = st.sidebar.button('EUR-USDを予測する')

#データ取得ボタンを押すとローソク足が表示される
if show_data:
    # 抽出期間を指定
    d = dt.datetime.today()
    y = d - dt.timedelta(days=5)
    end_date = d.strftime('%Y-%m-%d')
    start_date = y.strftime('%Y-%m-%d')

    # Yahooファイナンスより通貨情報を取得（抽出期間指定あり）
    data = pdr.get_data_yahoo(brandCode, end=end_date, start=start_date, interval='1H')

    # ローソク足の描画
    st.subheader('チャート')
    st.write('1時間あたりのローソク足が表示されます')
    st.write('yfinanceの日次データはOpen値とClose値が同一となっており、正確ではない可能性があるため。')
    fig = mpf.plot(data, type='candle')
    st.pyplot(fig)

#予測開始ボタンを押すと予測データの生成と結果表示を行う。
if get_data:
    # 抽出期間を指定
    d = dt.datetime.today()
    y = d - dt.timedelta(days=15)
    end_date = d.strftime('%Y-%m-%d')
    start_date = y.strftime('%Y-%m-%d')
    brandCode = 'JPY=X'

    # Yahooファイナンスより通貨情報を取得（抽出期間指定あり）
    data2 = pdr.get_data_yahoo(brandCode, end=end_date, start=start_date, interval='1H')
    data2 = data2.reset_index(drop= False)
    base = dt.time(8, 0, 0)
    data2 = data2[data2['Datetime'].dt.time == base]
    data2 = data2.drop(['Close', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
    data2 = data2.reset_index(drop= True)

    st.header('USD-JPY予測結果')
    st.write('過去10日間の日本時間17時時点の価格が表示されます')
    st.write('このアプリは過去10日間の日本時間17時の始値を使って、翌日17時の始値を予測します。')
    st.write('予測アプリはLSTM（中間層は50）は使用します。')
    st.write(data2)

    #入力するOpeを平均値が0、標準偏差が1になるように標準化を行う
    val = data2['Open'].values.reshape(-1, 1)
    scaler = StandardScaler()
    val_std = scaler.fit_transform(val)
    #予測データの実装
    val2 = []  # 入力データ(過去10日分)
    val2.append(val_std[-10:])
    #ndarrayに変換
    val3 = np.array(val2)
    val4 = torch.Tensor(val3)
    #st.write('予測用のInputデータはこちら')
    #st.write(val4)

    #モデルを定義
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.lstm = nn.LSTM(1, 50, batch_first=True,num_layers=1)
            self.linear = nn.Linear(50, 1)
        def forward(self, x):
            output, (hidden, cell) = self.lstm(x)
            output = self.linear(output[:, -1, :])
            return output

    # 使用するデバイスを設定
    device = torch.device('cpu')

    # 保存したモデルパラメータの読み込み
    model = Net().to(device)
    model.load_state_dict(torch.load('model50100.pkl'))
    model.eval()

    with torch.no_grad():
        y = model(val4)
    y_pred = y.cpu().data.numpy()
    y_pred = scaler.inverse_transform(y_pred)
    y_pred = y_pred[0][0]
    st.write('予測結果')
    st.write(y_pred)

#予測開始ボタンを押すと予測データの生成と結果表示を行う。
if get_data2:
    # 抽出期間を指定
    d = dt.datetime.today()
    y = d - dt.timedelta(days=15)
    end_date = d.strftime('%Y-%m-%d')
    start_date = y.strftime('%Y-%m-%d')
    brandCode = 'EUR=X'

    # Yahooファイナンスより通貨情報を取得（抽出期間指定あり）
    data2 = pdr.get_data_yahoo(brandCode, end=end_date, start=start_date, interval='1H')
    data2 = data2.reset_index(drop= False)
    base = dt.time(8, 0, 0)
    data2 = data2[data2['Datetime'].dt.time == base]
    data2 = data2.drop(['Close', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
    data2 = data2.reset_index(drop= True)

    st.header('EUR-USD予測結果')
    st.write('過去10日間の日本時間17時時点の価格が表示されます')
    st.write('このアプリは過去10日間の日本時間17時の始値を使って、翌日17時の始値を予測します。')
    st.write('予測アプリはLSTM（中間層は50）は使用します。')
    st.write(data2)

    #入力するOpeを平均値が0、標準偏差が1になるように標準化を行う
    val = data2['Open'].values.reshape(-1, 1)
    scaler = StandardScaler()
    val_std = scaler.fit_transform(val)
    #予測データの実装
    val2 = []  # 入力データ(過去10日分)
    val2.append(val_std[-10:])
    #ndarrayに変換
    val3 = np.array(val2)
    val4 = torch.Tensor(val3)
    #st.write('予測用のInputデータはこちら')
    #st.write(val4)

    #モデルを定義
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.lstm = nn.LSTM(1, 50, batch_first=True,num_layers=1)
            self.linear = nn.Linear(50, 1)
        def forward(self, x):
            output, (hidden, cell) = self.lstm(x)
            output = self.linear(output[:, -1, :])
            return output

    # 使用するデバイスを設定
    device = torch.device('cpu')

    # 保存したモデルパラメータの読み込み
    model = Net().to(device)
    model.load_state_dict(torch.load('model50100eur.pkl'))
    model.eval()

    with torch.no_grad():
        y = model(val4)
    y_pred = y.cpu().data.numpy()
    y_pred = scaler.inverse_transform(y_pred)
    y_pred = y_pred[0][0]
    st.write('予測結果')
    st.write(y_pred)





  
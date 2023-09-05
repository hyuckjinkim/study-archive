import numpy as np
import pandas as pd

import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class StockDataLoader:
    def __init__(
        self,ticker,start_date=None,end_date=None,
        test_size=0.2,validation_size=0.2,seed=0,
        scaling=True,target_logscale=False,lag_length=3,
    ):
        
        # (1) data load
        self.data = self._load(ticker)
        
        # (2) preprocessing
        self.data = self._preprocessing(self.data,lag_length)
        if start_date is not None:
            self.data = self.data[self.data.date>=start_date]
        if end_date is not None:
            self.data = self.data[self.data.date<=end_date]
        self.data.reset_index(drop=True,inplace=True)
        
        # (3) train test split
        self.train_data, self.validation_data, self.test_data = self._train_validation_test_split(
            self.data,test_size=test_size,validation_size=validation_size,seed=seed)
        
        # (4) scaling
        if scaling:
            self.train_data, self.validation_data, self.test_data = self._scaling(
                self.train_data,
                self.validation_data,
                self.test_data,
                feature_range=(0,1),
                ignore_features=['date','adj_close']
            )
        
        # (5) final dataset
        # target -> move last column
        self.train_data = pd.concat([
            self.train_data.drop('adj_close',axis=1),
            self.train_data['adj_close']
        ],axis=1)
        self.test_data  = pd.concat([
            self.test_data .drop('adj_close',axis=1),
            self.test_data ['adj_close']
        ],axis=1)
        # delete date column
        #self.train_data.drop('date',axis=1,inplace=True)
        #self.test_data .drop('date',axis=1,inplace=True)
        if target_logscale:
            self.train_data['adj_close'] = np.log(self.train_data['adj_close'])
            self.test_data ['adj_close'] = np.log(self.test_data ['adj_close'])
            
    def get_data(self):
        return self.train_data, self.validation_data, self.test_data
        
    def _load(self,ticker):
        df = fdr.DataReader(ticker,start=None,end=None).reset_index()
        df.columns = [col.lower().replace(' ','_') for col in df.columns]
        return df
        
    def _preprocessing(self,data,lag_length):
        df = data.copy()
        
        for w in sorted(df['date'].dt.weekday.unique()):
            df[f'weekday_{w}'] = np.where(df['date'].dt.weekday==w,1,0)
        for m in sorted(df['date'].dt.month.unique()):
            df[f'month_{m}'] = np.where(df['date'].dt.month==m,1,0)
        df['year'] = df['date'].dt.year
        
        for lag_i in range(lag_length):
            df[f'adj_close_lag_{lag_i+1}'] = df['adj_close'].shift(lag_i+1)
        
        # (1) 수정종가 사용 -> 종가 제외
        df = df.drop('close',axis=1)
        
        # (2) 등략률 : 전일자 기준 상승률 -> (오늘일자 - 전일자) / 전일자
        df['close_change']  = 100 * (df['adj_close'] - df['adj_close'].shift(1)) / df['adj_close'].shift(1)
        df['volume_change'] = 100 * (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
        
        # (3) 이동평균, 이동표준편차, 볼린저밴드
        aggregate_sizes = [5,10,20,60,120]
        for s in aggregate_sizes:
            # 이동평균, 이동표준편차
            df[f'close_ma_{s}']  = df['adj_close'].rolling(s).mean() # N일 이동평균
            df[f'close_sd_{s}']  = df['adj_close'].rolling(s).std()  # N일 이동표준편차
            df[f'volume_ma_{s}'] = df['volume'].rolling(s).mean()
            df[f'volume_sd_{s}'] = df['volume'].rolling(s).std()
            # 볼린저밴드
            df[f'bollinger_band_upper_{s}'] = df[f'close_ma_{s}'] + 2*df[f'close_sd_{s}'] # 상단밴드
            df[f'bollinger_band_lower_{s}'] = df[f'close_ma_{s}'] - 2*df[f'close_sd_{s}'] # 상단밴드

        # (4) MACD(Moving Average Convergence  and Divergence)
        #     - macd_short, macd_long, macd_signal = 12,26,9
        macd_short, macd_long, macd_signal = 12, 26, 9
        df['macd_short']  = df['adj_close'].rolling(macd_short).mean()
        df['macd_long']   = df['adj_close'].rolling(macd_long).mean()
        df['macd']        = df['macd_short']-df['macd_long']
        df['macd_signal'] = df['macd'].rolling(macd_signal).mean()  
        df['macd_sign']   = np.where(df['macd']>df['macd_signal'],1,0)

        # (5) N주 평균 등락률
        aggregate_sizes = [1,3,6,13,26,52]
        for s in aggregate_sizes:
            df[f'mean_close_change_{s}week']  = df['close_change'] .rolling(s*5).mean()
            df[f'mean_volume_change_{s}week'] = df['volume_change'].rolling(s*5).mean()
            
        return df
    
    def _train_validation_test_split(self,data,test_size=0.2,validation_size=0.2,seed=0):
        
        validation_size = int(len(data) * validation_size)
        test_size       = int(len(data) * test_size)
        train_size      = len(data)-validation_size-test_size

        train_data      = data[:train_size]
        validation_data = data[train_size:train_size+validation_size]
        test_data       = data[train_size+validation_size:]
        
        return train_data, validation_data, test_data

    def _scaling(self,train_data,validation_data,test_data,feature_range=(0,1),ignore_features=[]):
        tr_data, va_data, te_data = train_data.copy(), validation_data.copy(), test_data.copy()
        features = [col for col in train_data.columns if col not in ignore_features]
        
        # 데이터 스케일링
        scaler = MinMaxScaler(feature_range=feature_range)
        tr_data[features] = scaler.fit_transform(tr_data[features])
        va_data[features] = scaler.transform(va_data[features])
        te_data[features] = scaler.transform(te_data[features])
        
        return tr_data,va_data,te_data
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import datetime

def Simulation(TICKER,SIM_START_DATE,SIM_END_DATE,FIGURE=True):

    #TICKER = ['SPY','AAPL','TQQQ'][1]
    #print(TICKER)

    #SIM_START_DATE = '2018-01-01'
    #SIM_END_DATE = '2022-12-31'
    BUY_RATE = 0.2
    CHARGE = 0.015/100
    
    #-----------------------------------------------------------------------------------------------------#
    # 1. Data Load
    #-----------------------------------------------------------------------------------------------------#
    
    # (1) ksdkrw data
    usdkrw_df = yf.download(['USDKRW=X','JPYKRW=X'],progress=False)
    usdkrw_df = usdkrw_df['Adj Close']['USDKRW=X']\
        .reset_index()\
        .rename(columns={'Date':'date','USDKRW=X':'usdkrw'})
    # display(usdkrw_df.head())

    start_date = str(usdkrw_df.date.min())[:10]
    end_date   = str(usdkrw_df.date.max())[:10]

    # (2) spy data
    spy_df = yf.download(TICKER,start=start_date,end=end_date,progress=False)
    spy_df = spy_df.reset_index()
    spy_df.columns = [col.lower().replace(' ','_') for col in spy_df.columns]
    spy_df = spy_df[['date','adj_close']]
    # display(spy_df.head())

    # (3) join
    df = pd.merge(spy_df,usdkrw_df,how='left',on='date')
    df.bfill(inplace=True)
    # display(df.head())
    
    # display(df.isnull().sum())
    # display(df.head())
    
    #-----------------------------------------------------------------------------------------------------#
    # 2. Set Simulation
    #-----------------------------------------------------------------------------------------------------#
    
    # 월평균임금 : https://www.index.go.kr/unify/idx-info.do?pop=1&idxCd=5032
    # 단위 : 천원
    month_sales = np.array([2326,2428,2527,2617,2700,2740,2833,2896,3028,3138,3180,3271]) / 10
    month_sales_inc_rate = pd.Series(month_sales) / pd.Series(month_sales).shift(1) - 1

    # # 월평균임금 상승비율 평균
    # round(month_sales_inc_rate.mean(),3)
    
    # (1) simulation date filter
    start_loc = df.date>=datetime.datetime.strptime(SIM_START_DATE,'%Y-%m-%d')
    end_loc   = df.date<=datetime.datetime.strptime(SIM_END_DATE,'%Y-%m-%d')
    df2 = df[start_loc & end_loc].reset_index(drop=True)
    df2['year']  = df2.date.dt.year
    df2['month'] = [str(month).zfill(2) for month in df2.date.dt.month]
    df2['yyyymm'] = df2.year.astype(str)+df2.month.astype(str)

    # (2) get first date in each month
    df3 = df2.groupby('yyyymm')[['year','date','adj_close','usdkrw']].first().reset_index()

    # (3) month sales data
    month_sales_df = df3\
        .date.dt.year\
        .drop_duplicates()\
        .reset_index(drop=True)\
        .to_frame()\
        .rename(columns={'date':'year'})
    month_sales_df['month_sales'] = [232.6*(1+month_sales_inc_rate.mean())**i for i in range(len(month_sales_df))]

    # (4) join
    df4 = pd.merge(df3,month_sales_df,how='left',on='year').drop(columns=['year','yyyymm'])
    
    # (5) simulation
    sim_list = []
    remaining_dollar = 0
    for date,adj_close,usdkrw,month_sales in df4[['date','adj_close','usdkrw','month_sales']].values:
        my_dollar = remaining_dollar + (month_sales * BUY_RATE * 10000 / usdkrw)              # 남은돈 + 월급의20%
        n_buy_stock = np.floor(my_dollar/(adj_close*(1+CHARGE)))                              # 수수료를 고려해서 살 수 있는 SPY의 수
        buy_dollar = adj_close*(1+CHARGE) * n_buy_stock                                       # 수수료를 고려해서 사는 돈(달러)
        remaining_dollar = my_dollar - buy_dollar                                             # 남는돈 update
        sim_list.append([date,month_sales,n_buy_stock,adj_close,buy_dollar,remaining_dollar]) # append

    sim_df = pd.DataFrame(sim_list,columns=['date','month_sales','n_buy_stock','stock_price','buy_dollar','remaining_dollar'])

    asis_price = sum(sim_df.n_buy_stock * sim_df.stock_price) * df.tail(1).usdkrw.values[0]
    tobe_price = sum(sim_df.n_buy_stock) * df.tail(1).adj_close.values[0] * df.tail(1).usdkrw.values[0]
    earn_price = tobe_price - asis_price
    earn_rate  = tobe_price / asis_price

    # (7) result print
    print('> 시뮬레이션 기간  : {} ~ {} ({} Year)'.format(str(sim_df.date.min())[:10],str(sim_df.date.max())[:10],sim_df.date.dt.year.max()-sim_df.date.dt.year.min()+1))
    print('> 보유한 SPY의 수  : {:.0f}'.format(sum(sim_df.n_buy_stock)))
    print('> 구매 총가격 ($)  : {:.3f}'.format(sum(sim_df.n_buy_stock * sim_df.stock_price)))
    print('> 구매 평단가 ($)  : {:.3f}'.format(sum(sim_df.n_buy_stock * sim_df.stock_price) / sum(sim_df.n_buy_stock)))
    print('')
    print('> 구매 총가격 (₩)  : {:,}'.format(int(asis_price)))
    print('> 판매 총가격 (₩)  : {:,}'.format(int(tobe_price)))
    print('> 이익 (₩)         : {:,}'.format(int(earn_price)))
    print('> 이익비           : {:.3f}'.format(earn_rate))
    print('')

    # (8) graph
    if FIGURE:
        date_list = pd.date_range(df.date.min(),df.date.max())
        xticks = [date for date in date_list if (date.day==1) & (date.month==1)]
        xticklabels = [str(xtick)[:7] for xtick in xticks]

        fig,ax = plt.subplots(figsize=(15,7))

        sns.lineplot(x=df.date,y=df.adj_close,ax=ax,color='black',alpha=0.7) # (1) spy stock price
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels,rotation=90)
        ax.grid()

        ax2 = ax.twinx()
        sns.lineplot(x=df.date,y=df.usdkrw,ax=ax2,color='gray',alpha=0.7)    # (2) USDKRW

        ax3 = ax.twinx()
        sns.lineplot(x=sim_df.date,y=sim_df.n_buy_stock,ax=ax3,linestyle='--')
        ax3.set_yticks([])
        ax3.set_ylabel('')

        for value in [datetime.datetime.strptime(SIM_START_DATE,'%Y-%m-%d'), df.tail(1).date.values[0]]:
            plt.axvline(value,color='red',linestyle='--')

        fig.legend(loc=2,labels=['adj_close','usdkrw','n_buy_stock'],bbox_to_anchor=(0.95,0.9))
        plt.show()
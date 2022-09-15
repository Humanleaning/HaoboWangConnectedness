# 导入各种库
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
#-------
from models.RIXVaR.InfoExtractor import IvyDBUS_InfoExtractor
from models.RIXVaR.InfoExtractor import IvyDBEurope_InfoExtractor
from models.RIXVaR.ComputeVolVaR import compute_movments_var_spline




#%%
market = 'AEX'
url = r'data/Index/'
if market == 'SPX':
    with open(url+market+'/Style1.pkl', 'rb') as file:
        Options, Spot, Forward, Interest = pickle.load(file)
    VaRandVol = pd.DataFrame()  # 构建一个数据框，用来存储数据
    Dates = Options.index.unique()
    alphas = [0.01, 0.05, 0.1]
    info_ex = IvyDBUS_InfoExtractor(Options, Forward, Interest, Spot)
else:
    with open(url+market+'/Style1.pkl', 'rb') as file:
        Options, Spot, Interest, Dividend, Forward = pickle.load(file)
    VaRandVol = pd.DataFrame()  # 构建一个数据框，用来存储数据
    Dates = Options.index.unique()
    alphas = [0.01, 0.05, 0.1]
    info_ex = IvyDBEurope_InfoExtractor(market, Options, Forward, Dividend, Interest, Spot)

Dates = [pd.to_datetime('2010-2-16'), pd.to_datetime('2010-2-18')]
#%%
for date in tqdm(Dates):
   VaRandVol = compute_movments_var_spline(info_ex, date, VaRandVol)

#%%
VaRandVol['VaR_30_0.01_ln'] = np.log(VaRandVol['VaR_30_0.01'] / VaRandVol.S)
VaRandVol['VaR_30_0.05_ln'] = np.log(VaRandVol['VaR_30_0.05'] / VaRandVol.S)
VaRandVol['VaR_30_0.1_ln'] = np.log(VaRandVol['VaR_30_0.1'] / VaRandVol.S)
VaRandVol = VaRandVol[['S',
                       'VaR_30_0.01_ln',
                       'VaR_30_0.05_ln',
                       'VaR_30_0.1_ln',
                       'VaR_30_0.05',
                       'VaR_30_0.01',
                       'VaR_30_0.05',
                       'VaR_30_0.1',
                       'vol',
                       'skew',
                       'kurto',
                       'miu',
                       'sigma',
                       'eta']]
# Fulltime = pd.DataFrame(index=pd.date_range('2008-1-1','2021-12-31'))
# Fulltime.index.name = 'date'
# VaRandVol.sort_index(inplace=True)
# VaRandVol.index.name = 'date'
# VaRandVolResult = pd.merge(Fulltime, VaRandVol, how='left', left_on='date', right_on='date')
# VaRandVolResult.to_csv('SPX.csv')
VaRandVol.to_csv('SPX.csv')
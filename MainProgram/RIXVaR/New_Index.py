# 导入各种库
import pandas as pd
import pickle
from tqdm import tqdm
#-------
from models.RIXVaR.InfoExtractor import IvyDBUS_InfoExtractor
from models.RIXVaR.InfoExtractor import IvyDBEurope_InfoExtractor
from models.RIXVaR.ComputeRIX import compute_rix_spline
# import importlib
# import models.ComputeRIX
# importlib.reload(models.ComputeRIX)
# from models.ComputeRIX import compute_rix_spline

#%%
market = 'SPX'
url = r'data/Index/'
if market == 'SPX':
    with open(url+market+'/Style1.pkl', 'rb') as file:
        Options, Spot, Forward, Interest = pickle.load(file)
    Dates = Options.index.unique()
    alphas = [0.01, 0.05, 0.1]
    info_ex = IvyDBUS_InfoExtractor(Options, Forward, Interest, Spot)

    VaRandVol = pd.read_csv(url+market+'/Otherdata.csv', parse_dates=['date'], index_col=0)

else:
    with open(url+market+'/Style1.pkl', 'rb') as file:
        Options, Spot, Interest, Dividend, Forward = pickle.load(file)
    Dates = Options.index.unique()
    alphas = [0.01, 0.05, 0.1]
    info_ex = IvyDBEurope_InfoExtractor(market, Options, Forward, Dividend, Interest, Spot)

    VaRandVol = pd.DataFrame()  # 构建一个数据框，用来存储数据

#Dates = [pd.to_datetime('2010-2-16'), pd.to_datetime('2010-2-18')]

#%%
import time
time_start=time.time()
for date in tqdm(Dates[0:10]):
   VaRandVol = compute_rix_spline(info_ex, date, VaRandVol, precision=1000)
time_end=time.time()
print('time cost', time_end-time_start, 's')

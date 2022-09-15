# 导入各种库
import pandas as pd
import pickle
from tqdm import tqdm
#-------
from models.RIXVaR.InfoExtractor import IvyDBUS_InfoExtractor
from models.RIXVaR.InfoExtractor import IvyDBEurope_InfoExtractor
from models.RIXVaR.ComputeVolVaR import compute_movments_var_spline
from models.RIXVaR.ComputeRIX import compute_rix_kernel

import multiprocessing as mp
#%%
def job(market):

    if market in ['AEX', 'CAC 40', 'DAX', 'FTSE MIB', 'FTSE 100', 'SPX']:
        url = r'data/Index/'
        if market == 'SPX':
            with open(url + market + '/Style1.pkl', 'rb') as file:
                Options, Spot, Forward, Interest = pickle.load(file)
            Dates = Options.index.unique()
            info_ex = IvyDBUS_InfoExtractor(Options, Forward, Interest, Spot)
        else:
            with open(url + market + '/Style1.pkl', 'rb') as file:
                Options, Spot, Interest, Dividend, Forward = pickle.load(file)

            Dates = Options.index.unique()
            info_ex = IvyDBEurope_InfoExtractor(market, Options, Forward, Dividend, Interest, Spot)


        VaRandVol = pd.read_csv(url+market+'/'+market+'.csv', parse_dates=['date'], index_col=0)
        for date in tqdm(Dates):
            VaRandVol = compute_movments_var_spline(info_ex, date, VaRandVol)

        VaRandVol.to_csv(url + market + '/'+market+'RIX_GEV.csv')

    else:
        url = r'data/ETF/'
        with open(url + market + '/Style1.pkl', 'rb') as file:
            Options, Spot, Forward, Interest = pickle.load(file)
        VaRandVol = pd.DataFrame()  # 构建一个数据框，用来存储数据
        Dates = Options.index.unique()
        info_ex = IvyDBUS_InfoExtractor(Options, Forward, Interest, Spot)


        VaRandVol = pd.read_csv(url + market + '/' + market + '.csv', parse_dates=['date'], index_col=0)
        for date in tqdm(Dates):
            VaRandVol = compute_rix_kernel(info_ex, date, VaRandVol, precision=1000)

        VaRandVol.to_csv(url + market + '/'+market+'RIX_GEV.csv')

#%%
MarketsIndex = ['AEX', 'CAC 40', 'DAX', 'FTSE MIB', 'FTSE 100', 'SPX']
MarketsETF = ['EWZ', 'FXI', 'RSX', 'EWW', 'EWY', 'INDA', 'KSA'
              'SLV', 'USO', 'GLD',
              'XLB', 'XLC', 'XLE', 'XLF', 'XLI', "XLK", 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
Markets = MarketsIndex + MarketsETF
#%%
def multicore():
    pool = mp.Pool(len(Markets))
    res = pool.map(job, Markets)
    return res
#%%
if __name__ == '__main__':
    res = multicore()
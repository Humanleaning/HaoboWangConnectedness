# 导入各种库
import pandas as pd
import pickle
from tqdm import tqdm
#-------
from models.RIXVaR.InfoExtractor import IvyDBUS_InfoExtractor
from models.RIXVaR.ComputeRIX import compute_rix_kernel

# %%
market = 'USO'
url = r'data/ETF/'
with open(url + market + '/Style1.pkl', 'rb') as file:
    Options, Spot, Forward, Interest = pickle.load(file)
# %%
VaRandVol = pd.read_csv(url+market+'/Otherdata.csv', parse_dates=['date'], index_col=0)  # 构建一个数据框，用来存储数据
Dates = Options.index.unique()
#alphas = [0.01, 0.05, 0.1]
#%%
info_ex = IvyDBUS_InfoExtractor(Options, Forward, Interest, Spot)
#%% GEV分布的版本
for date in tqdm(Dates):
    VaRandVol = compute_rix_kernel(info_ex, date, VaRandVol, precision=1000)

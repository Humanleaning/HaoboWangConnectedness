# 导入各种库
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from models.RIXVaR.PolationTechnics import LLkernel_L
from models.RIXVaR.RiskNeutralDensityComputer import VaRComputer_p
from models.RIXVaR.GEV import GEV_computer
#-------
from models.RIXVaR.InfoExtractor import IvyDBUS_InfoExtractor
from models.RIXVaR.ComputeVolVaR import compute_movments_var_kernel


# %%
market = 'USO'
url = r'data/ETF/'
with open(url + market + '/Style1.pkl', 'rb') as file:
    Options, Spot, Forward, Interest = pickle.load(file)
# %%
VaRandVol = pd.DataFrame()  # 构建一个数据框，用来存储数据
Dates = Options.index.unique()
alphas = [0.01, 0.05, 0.1]
#%%
info_ex = IvyDBUS_InfoExtractor(Options, Forward, Interest, Spot)
#%% GEV分布的版本
for date in tqdm(Dates[0:100]):
    VaRandVol = compute_movments_var_kernel(info_ex, date, VaRandVol)
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
Fulltime = pd.DataFrame(index=pd.date_range('2008-1-1','2021-12-31'))
Fulltime.index.name = 'date'
VaRandVol.sort_index(inplace=True)
VaRandVol.index.name = 'date'
VaRandVolResult = pd.merge(Fulltime, VaRandVol, how='left', left_on='date', right_on='date')
VaRandVolResult.to_csv(url+market+'/llkernel_GEV.csv')

#%%
def VisualRN(datetime=pd.to_datetime('2008-7-16'), eps=0.01, left=0.5, right=1.2):
    date = datetime
    info = info_ex.getoneday(date)
    if info[-1]:  # 进入两个插值的模式
        Option, S, F, r, tenor_front, tenor_rear, insert = info
        VaRandVol.loc[date, 'S'] = S

        chine_front = Option.loc[(Option.tenor == tenor_front) ]
        chine_rear = Option.loc[(Option.tenor == tenor_rear) ]
        kr1 = LLkernel_L(chine_front.impl_volatility, chine_front.strike_price,eps=0.5)
        kr2 = LLkernel_L(chine_rear.impl_volatility, chine_rear.strike_price,eps=0.5)
        vc = VaRComputer_p(30, S, F, r,  insert=True, args=(kr1, kr2, tenor_front, tenor_rear))

        num = 50
        strike_grid = np.linspace(left * F, right * F, num)
        pdf_grid = np.zeros(strike_grid.shape)
        cdf_grid = np.zeros(strike_grid.shape)
        iv_grid = np.zeros(strike_grid.shape)
        for i in range(num):
            iv_grid[i], cdf_grid[i], pdf_grid[i] = vc.comp_rn(strike_grid[i])

        fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=False, sharey=False)
        axes[0].scatter(strike_grid, iv_grid)
        axes[0].set_title('iv')
        axes[0].axvline(kr1.K_left)
        axes[0].axvline(kr2.K_left)

        axes[1].plot(strike_grid, cdf_grid)
        axes[1].set_title("cdf")
        axes[1].axvline(kr1.K_left)
        axes[1].axvline(kr2.K_left)

        axes[2].plot(strike_grid, pdf_grid)
        axes[2].set_title('pdf')
        axes[2].axvline(kr1.K_left)
        axes[2].axvline(kr2.K_left)
        plt.title(date)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.show()


    else:  # 进入直接有一个30天的模式
        Option, S, F, r, tenor, insert = info
        VaRandVol.loc[date, 'S'] = S

        chine = Option.loc[(Option.tenor == 30) & (Option.moneynessF <= 1)]
        kr = LLkernel_L(chine.impl_volatility, chine.strike_price,eps=0.5)
        vc = VaRComputer_p(tenor, S, F, r, insert=False, args=kr)
        num = 50
        strike_grid = np.linspace(left * F, right * F, num)
        pdf_grid = np.zeros(strike_grid.shape)
        cdf_grid = np.zeros(strike_grid.shape)
        iv_grid = np.zeros(strike_grid.shape)
        for i in range(num):
            iv_grid[i], cdf_grid[i], pdf_grid[i] = vc.comp_rn(strike_grid[i])

        fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=False, sharey=False)
        axes[0].scatter(strike_grid, iv_grid)
        axes[0].set_title('iv')
        axes[0].axvline(kr.K_left)

        axes[1].plot(strike_grid, cdf_grid)
        axes[1].set_title("cdf")
        axes[1].axvline(kr.K_left)

        axes[2].plot(strike_grid, pdf_grid)
        axes[2].set_title('pdf')
        axes[2].axvline(kr.K_left)
        plt.title(date)
        plt.show()
#%%
VisualRN(pd.to_datetime('2021-5-14'),eps=1,left=0.2,right=1.2)
#%%
def VisualRN_GEV(datetime=pd.to_datetime('2008-7-16'), eps=0.01, left=0.5, right=1.2):
    date = datetime
    info = info_ex.getoneday(date)
    if info[-1]:  # 进入两个插值的模式
        Option, S, F, r, tenor_front, tenor_rear, insert = info
        VaRandVol.loc[date, 'S'] = S

        chine_front = Option.loc[(Option.tenor == tenor_front) ]
        chine_rear = Option.loc[(Option.tenor == tenor_rear) ]
        kr1 = LLkernel_L(chine_front.impl_volatility, chine_front.strike_price,eps=0.5)
        kr2 = LLkernel_L(chine_rear.impl_volatility, chine_rear.strike_price,eps=0.5)
        vc = VaRComputer_p(S, F, r,  insert=True, args=(kr1, kr2, tenor_front, tenor_rear))

        if kr1.K_left != kr2.K_left:
            # X0L = max(kr1.K_left, kr2.K_left)
            # X1L = min(kr1.K_left, kr2.K_left)
            # ignore0, ALPHA0L, fX0L = vc.comp_rn(X0L)
            # ignore1, ALPHA1L, fX1L = vc.comp_rn(X1L)
            X1L = max(kr1.K_left, kr2.K_left)+kr1.eps
            ignore1, ALPHA1L, fX1L = vc.comp_rn(X1L)
            ALPHA0L = ALPHA1L + 0.03
            X0L = vc.comp_VaR(ALPHA0L)
            ignore0, ALPHA0L, fX0L = vc.comp_rn(X0L)
        else:
            X1L = max(kr1.K_left, kr2.K_left)
            ignore1, ALPHA1L, fX1L = vc.comp_rn(X1L)
            ALPHA0L = ALPHA1L + 0.03
            X0L = vc.comp_VaR(ALPHA0L)
            ignore0, ALPHA0L, fX0L = vc.comp_rn(X0L)

        gev = GEV_computer(X0L, fX0L, ALPHA0L, X1L, fX1L, (F, 0.15 * F, 0.1), shift=2 * round(F))

        num = 50
        strike_grid = np.linspace(left * F, right * F, num)
        pdf_grid = np.zeros(strike_grid.shape)
        cdf_grid = np.zeros(strike_grid.shape)
        iv_grid = np.zeros(strike_grid.shape)
        iv_front_grid = np.zeros(strike_grid.shape)
        iv_rear_grid = np.zeros(strike_grid.shape)
        for i in range(num):
            iv_grid[i], cdf_grid[i], pdf_grid[i] = vc.comp_rn(strike_grid[i])
            if cdf_grid[i]<ALPHA0L:
                cdf_grid[i] = gev.cdf(strike_grid[i])
                pdf_grid[i] = gev.pdf(strike_grid[i])
            iv_front_grid[i] = kr1.fit(strike_grid[i])
            iv_rear_grid[i] = kr2.fit(strike_grid[i])

        fig, axes = plt.subplots(5, 1, figsize=(6, 20), sharex=False, sharey=False)
        axes[0].scatter(strike_grid, iv_grid)
        axes[0].set_title('iv')
        axes[0].axvline(kr1.K_left,c='r')
        axes[0].axvline(kr2.K_left)
        axes[0].axvline(F,c='c')

        axes[1].plot(strike_grid, cdf_grid)
        axes[1].set_title("cdf")
        axes[1].axvline(kr1.K_left,c='r')
        axes[1].axvline(kr2.K_left)
        #axes[1].axhline(0.01)
        axes[1].axhline(0.05)
        axes[1].axhline(0.10)
        axes[1].axvline(F, c='c')
        #start, end = ax.get_xlim()
        # axes[1].yaxis.set_ticks(np.arange(0.01, round(max(cdf_grid),2)-0.4, 0.02))
        # axes[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))

        axes[2].plot(strike_grid, pdf_grid)
        axes[2].set_title('pdf')
        axes[2].axvline(kr1.K_left,c='r')
        axes[2].axvline(kr2.K_left)
        axes[2].axvline(F, c='c')
        axes[2].axvline(X0L, c='c')

        axes[3].plot(strike_grid, iv_front_grid)
        axes[3].axvline(kr1.K_left, c='r')
        axes[3].axvline(F, c='c')
        axes[3].scatter(chine_front.strike_price, chine_front.impl_volatility, s=5, c='r')

        axes[4].plot(strike_grid, iv_rear_grid)
        axes[4].axvline(kr2.K_left)
        axes[4].axvline(F, c='c')
        axes[4].scatter(chine_rear.strike_price, chine_rear.impl_volatility, s=5, c='r')

        plt.title(date)
        #plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.show()


    else:  # 进入直接有一个30天的模式
        Option, S, F, r, tenor, insert = info
        VaRandVol.loc[date, 'S'] = S

        chine = Option.loc[(Option.tenor == 30) & (Option.moneynessF <= 1)]
        kr = LLkernel_L(chine.impl_volatility, chine.strike_price,eps=0.5)
        vc = VaRComputer_p(tenor, S, F, r, insert=False, args=kr)
        num = 50
        strike_grid = np.linspace(left * F, right * F, num)
        pdf_grid = np.zeros(strike_grid.shape)
        cdf_grid = np.zeros(strike_grid.shape)
        iv_grid = np.zeros(strike_grid.shape)
        for i in range(num):
            iv_grid[i], cdf_grid[i], pdf_grid[i] = vc.comp_rn(strike_grid[i])
            if cdf_grid[i]<ALPHA0L:
                cdf_grid[i] = gev.cdf(strike_grid[i])
                pdf_grid[i] = gev.pdf(strike_grid[i])
        fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=False, sharey=False)
        axes[0].scatter(strike_grid, iv_grid)
        axes[0].set_title('iv')
        axes[0].axvline(kr.K_left)

        axes[1].plot(strike_grid, cdf_grid)
        axes[1].set_title("cdf")
        axes[1].axvline(kr.K_left)

        axes[2].plot(strike_grid, pdf_grid)
        axes[2].set_title('pdf')
        axes[2].axvline(kr.K_left)
        plt.title(date)
        plt.show()

        # ALPHA0L, ALPHA1L, kr1.K_left, kr2.K_left,vc.comp_rn(3.1)[1],X1L,X0L
    return chine_rear
#%%
chine_rear = VisualRN_GEV(pd.to_datetime('2021-8-16'),eps=1,left=0.1,right=1.2)
#%%
# %% 线性插值的版本
# for date in tqdm(Dates[0:100]):
#     info = info_ex.getoneday(date)
#     if info[-1]:# 进入两个插值的模式
#         Option, S, F, r, tenor_front, tenor_rear, insert = info
#         VaRandVol.loc[date, 'S'] = S
#         # try:
#         chine_front = Option.loc[(Option.tenor == tenor_front) & (Option.moneynessF <= 1)]
#         chine_rear = Option.loc[(Option.tenor == tenor_rear) & (Option.moneynessF <= 1)]
#         kr1 = LLkernel_L(chine_front.impl_volatility, chine_front.strike_price)
#         kr2 = LLkernel_L(chine_rear.impl_volatility, chine_rear.strike_price)
#         vc = VaRComputer_p(30, S, F, r,  insert=True, args=(kr1, kr2, tenor_front, tenor_rear))
#
#         for alpha in alphas:
#             VaRandVol.loc[date, ('VaR_30_' + str(alpha))] = vc.comp_VaR(alpha)
#         # except:
#         #     for alpha in alphas:
#         #         VaRandVol.loc[date, ('VaR_30_' + str(alpha))] = 0
#
#     else:# 进入直接有一个30天的模式
#         Option, S, F, r, tenor, insert = info
#         VaRandVol.loc[date, 'S'] = S
#         # try:
#         chine = Option.loc[(Option.tenor == 30) & (Option.moneynessF <= 1)]
#         kr = LLkernel_L(chine.impl_volatility, chine.strike_price)
#         vc = VaRComputer_p(tenor, S, F, r, insert=False, args=kr)
#
#         for alpha in alphas:
#             VaRandVol.loc[date, ('VaR_30_' + str(alpha))] = vc.comp_VaR(alpha)
#         # except:
#         #     for alpha in alphas:
#         #         VaRandVol.loc[date, ('VaR_30_' + str(alpha))] = 0
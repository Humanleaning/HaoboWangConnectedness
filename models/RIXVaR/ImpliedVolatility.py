#这里来实现
# 这里最终决定来说, 就是数据集使用最基础的数据, 然后再筛选, 相当于比ETF期权多了一步
    # 1)就先通过一个(strike_price, mid_price, cp_flag)的链得到一个r,F,(strike_price, iv, cp_flag)
    # 2)然后带入类似ETF的程序
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression #只需要一元线性回归, 且需分清有无截距, 优先选取速度更快的sklearn而不是statsmodel
import py_vollib.black.implied_volatility
import py_vollib_vectorized

# class IVcomputer(object):
#     def __init__(self, option_chine, S, tenor=30, dividend=True):
#         """
#         为option_chine增加一列impl_volatility,然后返回
#         :param option_chine:pandas.DataFrame 需要包含(cp_flag, strike_price, prices), 按照这个顺序排布, 命名可以随意
#         :returns self.new_chine, 包含(cp_flag, strike_price, prices, impl_volatility)
#         """
#         self.S = S
#         self.annualized_tenor =tenor/365
#         #rename
#         option_chine.columns = ['cp_flag', 'strike_price', 'prices']
#         #生成strike和两种price的对应
#         table = pd.pivot_table(option_chine, values='prices', index=['strike_price'], columns=['cp_flag'], aggfunc=np.sum)
#         table = table[(table != 0).all(1)]# pivot会把没有共同值的部分设置为0, 此处去除
#         if dividend:
#             # q != 0
#             # exp(−rT)F0 − exp(−rT)K = C(K, T) − P(K, T)
#             # F0 = S * exp[(r-q)T], 即 (F0/S)*exp(-rT) = exp(-qT), ln[(F0/S)*exp(-rT)]/(-T) = q, q = ln{[F0*exp(-rT)]/S}/(-T)
#             strikes =table.index.values.reshape(-1,1)
#             reg = LinearRegression().fit(strikes, (table.C-table.P))
#             slope, intercept = reg.coef_[0], reg.intercept_
#             self.r = np.log(-1*slope)/(-1*self.annualized_tenor)
#             self.F = intercept/(-1*slope)
#             self.q = np.log(intercept/self.S)/(-1*self.annualized_tenor)
#         else:
#             # q == 0
#             # S − exp(−rT)K = C(K, T) − P(K, T)
#             # exp(−rT)K = S - C(K, T) + P(K, T)
#             strikes = table.index.values.reshape(-1, 1)
#             reg = LinearRegression(fit_intercept=False).fit(strikes, (self.S-table.C+table.P))
#             slope = reg.coef_[0]
#             self.r = np.log(slope)/(-1*self.annualized_tenor)
#             self.F = self.S/slope
#             self.q = 0
#
#         option_chine['impl_volatility'] = py_vollib_vectorized.vectorized_implied_volatility_black(option_chine.prices,
#                                                                                                      self.F,
#                                                                                                      option_chine.strike_price,
#                                                                                                      self.r,
#                                                                                                      self.annualized_tenor,
#                                                                                                      option_chine.cp_flag.str.lower(),
#                                                                                                      return_as='numpy')
#         option_chine.dropna(axis=0, how='any', inplace=True)
#         self.new_chine = option_chine



def IVcomputer(option_chine, S, q, r, tenor=30):
    """
    为option_chine增加一列impl_volatility,然后返回
    :param option_chine:pandas.DataFrame 需要包含(cp_flag, strike_price, prices), 按照这个顺序排布, 命名可以随意
    :returns self.new_chine, 包含(cp_flag, strike_price, prices, impl_volatility)
    """
    S = S
    q = q
    r = r
    annualized_tenor = tenor / 365
    F = S*np.exp((r-q)*annualized_tenor)

    option_chine.columns = ['cp_flag', 'strike_price', 'prices']

    option_chine['impl_volatility'] = py_vollib_vectorized.vectorized_implied_volatility_black(option_chine.prices,
                                                                                                 F,
                                                                                                 option_chine.strike_price,
                                                                                                 r,
                                                                                                 annualized_tenor,
                                                                                                 option_chine.cp_flag.str.lower(),
                                                                                                 return_as='numpy')
    return option_chine.dropna(axis=0, how='any')






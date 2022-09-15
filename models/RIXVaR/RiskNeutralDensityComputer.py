"""
原理上期权价格price反映了为了状态价格分布的所有信息, 在BS框架下, price=BS(BSIV, args), BSIV=g(args)
BS()易于定义, 从而我们只需要给出g(),则可以通过args获得关于状态价格分布的所有信息
从而在原理上, 本程序的作用为输入g()和args, 返回状态价格分布的统计特征信息

这里g()有两种选择, 一种为g(K), 一种为g(K, tao), 当然也可以使用其他的插值方法如OptionsMetrics的方法, 但是他们的方法效果不太好
后一种g(K, tao)可以通过固定tao, 即在对象定义时固定tao信息即可
从而这里输入的g()都为g(K)
以对象的形式输入, 对象需要有.fit()方法, 即 iv = model.fit(K)
"""

from scipy.integrate import simps
from scipy.stats import norm
import numpy as np
from scipy.optimize import fminbound

def Blacks_P(K, tenor, F, r, iv):  # 日期都是以日为单位，需要转化
    tenor = tenor / 365
    d1 = ((np.log(F / K) + np.square(iv) * tenor / 2)) / (iv * np.sqrt(tenor))
    d2 = d1 - iv * np.sqrt(tenor)
    price = np.exp(-r * tenor) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return price
def Blacks_C(K, tenor, F, r, iv):  # 日期都是以日为单位，需要转化
    tenor = tenor / 365
    d1 = ((np.log(F / K) + np.square(iv) * tenor / 2)) / (iv * np.sqrt(tenor))
    d2 = d1 - iv * np.sqrt(tenor)
    price = np.exp(-r * tenor) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return price

class VaRComputer_p(object):
    """
    实现可以利用插值程序进行left VaR寻找
    市场信息:tenor, S, F, r
    曲面信息:model, 能够实现model.fit(K)→iv即可
    是否插值:insert
    逼近程度:eps
    """
    def __init__(self, S, F, r, tenor=30, insert: bool=True, eps=0.01, args=0):
        """
        :param tenor:
        :param S:
        :param F:
        :param r:
        :param insert:
        :param eps:
        :param args: 如果是需要insert,则输入tuple, 如果不需要, 则输入待估计的模型就行
        """
        self.tenor = tenor
        self.S = S
        self.F = F
        self.r = r
        self.eps = eps
        self.insert = insert
        if insert:
            self.model1, self.model2, tenor_left, tenor_right = args
            self.multi_left = (tenor_right - self.tenor) / (tenor_right - tenor_left)
            self.multi_right = (self.tenor - tenor_left) / (tenor_right - tenor_left)
            self.comp_diff = self.Diff_for_2
        else:
            self.model = args
            self.comp_diff = self.Diff_for_1

    def Diff_for_2(self, K, alpha):
        iv = self.multi_left * self.model1.fit(K) + self.multi_right * self.model2.fit(K)
        derivative = (Blacks_P(K + self.eps, self.tenor, self.F, self.r, iv) - Blacks_P(K - self.eps, self.tenor,self.F, self.r, iv)) / (2 * self.eps)
        CDF = np.exp(self.r * self.tenor / 365) * derivative
        return np.fabs(CDF - alpha)

    def Diff_for_1(self, K, alpha):
        iv = self.model.fit(K)
        derivative = (Blacks_P(K + self.eps, self.tenor, self.F, self.r, iv) - Blacks_P(K - self.eps, self.tenor,self.F, self.r, iv)) / (2 * self.eps)
        CDF = np.exp(self.r * self.tenor / 365) * derivative
        return np.fabs(CDF - alpha)

    def comp_VaR(self, CDF):
        b = 1
        a = 1 - 0.02
        count = 0
        while self.comp_diff(a * self.S, CDF) < self.comp_diff(b * self.S, CDF):
            a -= 0.02
            b -= 0.02
            count += 1
            if count >= 50:
                return 0
            else:
                continue
        b += 0.02

        return fminbound(self.comp_diff, a * self.S, b * self.S, args=(CDF,))

    def comp_rn(self, K):
        if self.insert:
            iv = self.multi_left * self.model1.fit(K) + self.multi_right * self.model2.fit(K)
        else:
            iv = self.model.fit(K)
        derivative1 = (Blacks_P(K + self.eps, self.tenor, self.F, self.r, iv) - Blacks_P(K - self.eps, self.tenor,self.F, self.r, iv)) / (2 * self.eps)
        CDF = np.exp(self.r * self.tenor / 365) * derivative1
        derivative2 = (Blacks_P(K + self.eps, self.tenor, self.F, self.r, iv) +
                       Blacks_P(K - self.eps, self.tenor, self.F, self.r, iv) -
                       2*Blacks_P(K, self.tenor, self.F, self.r, iv))/ np.square(self.eps)
        PDF = np.exp(self.r * self.tenor / 365) * derivative2

        return iv, CDF, PDF

class VaRComputer_c(object):
    """
    实现可以利用插值程序进行left VaR寻找
    市场信息:tenor, S, F, r
    曲面信息:model, 能够实现model.fit(K)→iv即可
    是否插值:insert
    逼近程度:eps
    """
    def __init__(self, S, F, r, tenor=30, insert: bool=True, eps=0.01, args=0):
        """
        :param tenor:
        :param S:
        :param F:
        :param r:
        :param insert:
        :param eps:
        :param args: 如果是需要insert,则输入tuple, 如果不需要, 则输入待估计的模型就行
        """
        self.tenor = tenor
        self.S = S
        self.F = F
        self.r = r
        self.eps = eps
        self.insert = insert
        if insert:
            self.model1, self.model2, tenor_left, tenor_right = args
            self.multi_left = (tenor_right - self.tenor) / (tenor_right - tenor_left)
            self.multi_right = (self.tenor - tenor_left) / (tenor_right - tenor_left)
            self.comp_diff = self.Diff_for_2
        else:
            self.model = args
            self.comp_diff = self.Diff_for_1

    def Diff_for_2(self, K, alpha):
        iv = self.multi_left * self.model1.fit(K) + self.multi_right * self.model2.fit(K)
        derivative = (Blacks_C(K + self.eps, self.tenor, self.F, self.r, iv) - Blacks_C(K - self.eps, self.tenor,self.F, self.r, iv)) / (2 * self.eps)
        CDF = np.exp(self.r * self.tenor / 365) * derivative + 1
        return np.fabs(CDF - alpha)

    def Diff_for_1(self, K, alpha):
        iv = self.model.fit(K)
        derivative = (Blacks_C(K + self.eps, self.tenor, self.F, self.r, iv) - Blacks_C(K - self.eps, self.tenor,self.F, self.r, iv)) / (2 * self.eps)
        CDF = np.exp(self.r * self.tenor / 365) * derivative + 1
        return np.fabs(CDF - alpha)

    def comp_VaR(self, CDF):
        a = 1
        b = 1 + 0.02
        count = 0
        while self.comp_diff(a * self.S, CDF) > self.comp_diff(b * self.S, CDF):
            a += 0.02
            b += 0.02
            count += 1
            if count >= 50:
                return 0
            else:
                continue
        a -= 0.02

        return fminbound(self.comp_diff, a * self.S, b * self.S, args=(CDF,))

    def comp_rn(self, K):
        if self.insert:
            iv = self.multi_left * self.model1.fit(K) + self.multi_right * self.model2.fit(K)
        else:
            iv = self.model.fit(K)
        derivative1 = (Blacks_C(K + self.eps, self.tenor, self.F, self.r, iv) - Blacks_C(K - self.eps, self.tenor,self.F, self.r, iv)) / (2 * self.eps)
        CDF = np.exp(self.r * self.tenor / 365) * derivative1 + 1
        derivative2 = (Blacks_C(K + self.eps, self.tenor, self.F, self.r, iv) +
                       Blacks_C(K - self.eps, self.tenor, self.F, self.r, iv) -
                       2*Blacks_C(K, self.tenor, self.F, self.r, iv))/ np.square(self.eps)
        PDF = np.exp(self.r * self.tenor / 365) * derivative2

        return iv, CDF, PDF

class MovementComputer(object):
    """
    市场信息:tenor, S, F, r
    曲面信息:model, 能够实现model.fit(K)→iv即可
    是否插值:insert
    """
    def __init__(self, S, F, r, tenor=30, insert: bool=True, args=0, range='default'):
        self.S = S
        self.F = F
        self.r = r
        self.tenor = tenor
        self.insert = insert
        if insert:
            self.model1, self.model2, tenor_left, tenor_right = args
            self.multi_left = (tenor_right - self.tenor) / (tenor_right - tenor_left)
            self.multi_right = (self.tenor - tenor_left) / (tenor_right - tenor_left)
            self.fit = self.combine
        else:
            self.fit = args.fit
        if range == 'default':
            self.start = 0.01
            self.end = 3*self.S
        else:
            self.start = range[0]
            self.end = range[1]


        self.V, self.W, self.X = self.comp_VWX()
        self.factor = np.exp(self.r * self.tenor / 365)
        self.miu = self.factor - 1 - self.factor * (self.V / 2 + self.W / 6 + self.X / 24)
        self.vol = np.sqrt(self.factor * self.V - np.square(self.miu))
        self.skew = (self.factor * self.W - 3 * self.factor * self.miu * self.V + 2 * np.power(self.miu, 3)) / np.power((self.factor * self.V - np.square(self.miu)), 3 / 2)
        self.kurto = (self.factor * self.X - 4 * self.miu * self.W * self.factor + 6 * self.factor * np.square(self.miu) * self.V - 3*np.power(self.miu, 4)) / np.square(self.factor * self.V - np.square(self.miu))

    

    def combine(self, K):
        return self.multi_left*self.model1.fit(K) + self.multi_right*self.model2.fit(K)

    def comp_V(self, K, p):
        value = 2 * (1 - np.log(K / self.S)) * np.power(K, -2) * p
        v = simps(value, K)
        return v
    def comp_W(self, K, p):
        value = (6 * np.log(K / self.S) - 3 * np.square(np.log(K / self.S))) * np.power(K, -2) * p
        v = simps(value, K)
        return v
    def comp_X(self, K, p):
        value = (12 * np.power(np.log(K / self.S), 2) - 4 * np.power(np.log(K / self.S), 3)) * np.power(K, -2) * p
        v = simps(value, K)
        return v

    def comp_VWX(self):
        strikes = np.linspace(self.start, self.end, num=1999, endpoint=True)
        strikes = np.insert(strikes, 1000, self.S)
        strikes = np.sort(strikes)
        ivs = np.zeros(strikes.shape)
        for i in range(2000):
            ivs[i] = self.fit(strikes[i])

        put_strike = strikes[strikes <= self.S]
        put_prices = Blacks_P(K=strikes[strikes <= self.S], tenor=self.tenor, F=self.F, r=self.r, iv=ivs[strikes <= self.S])
        call_strike = strikes[strikes >= self.S]
        call_prices = Blacks_C(K=strikes[strikes >= self.S], tenor=self.tenor, F=self.F, r=self.r, iv=ivs[strikes >= self.S])

        V_put  = self.comp_V(put_strike, put_prices)
        V_call = self.comp_V(call_strike, call_prices)
        W_put  = self.comp_W(put_strike, put_prices)
        W_call = self.comp_W(call_strike, call_prices)
        X_put  = self.comp_X(put_strike, put_prices)
        X_call = self.comp_X(call_strike, call_prices)
        V = V_call + V_put
        W = W_call + W_put
        X = X_call + X_put

        return V, W, X

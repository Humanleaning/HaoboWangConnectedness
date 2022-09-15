
from scipy.integrate import simps
from scipy.stats import norm
import numpy as np
from scipy import integrate

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


class GEV_RIX_Compute(object):

    def __init__(self, gev, X0L, S, F, r, insert: bool=True, args=0, tenor=30, precision=2000):

        self.gev = gev
        self.precision = precision
        self.statr_point = -(gev.mu - gev.sigma/gev.eta - gev.shift)
        self.break_point = X0L
        self.S = S
        self.tenor = tenor
        self.F = F
        self.r = r
        if insert:
            self.model1, self.model2, tenor_left, tenor_right = args
            self.multi_left = (tenor_right - self.tenor) / (tenor_right - tenor_left)
            self.multi_right = (self.tenor - tenor_left) / (tenor_right - tenor_left)
            self.fit = self.combine
        else:
            self.fit = args.fit


        self.RIX_L, self.V, self.IV = self.ComputeMeasure()

    def combine(self, K):
        return self.multi_left * self.model1.fit(K) + self.multi_right * self.model2.fit(K)

    def f_x(self, S, K):
        return (K-S)*self.gev.pdf(S)

    def comp_IV(self, K, p):
        value =  2 * np.power(K, -2) * p
        v = simps(value, K)
        return v
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

    def ComputeMeasure(self):

        strikes = np.linspace(self.statr_point, self.S, num=self.precision, endpoint=True)
        # strikes = np.insert(strikes, 1000, self.S)
        # strikes = np.sort(strikes)
        # ivs = np.zeros(strikes.shape)
        prices = np.zeros(strikes.shape)
        break_index = np.where(strikes >= self.break_point)[0][0]

        #prices[0] = 0
        for i in range(1, break_index):
            prices[i] = np.exp(-self.r * self.tenor / 365) * \
                        integrate.quad(self.f_x, self.statr_point, strikes[i], args=(strikes[i],))[0]

        ivs = np.zeros(strikes.shape)
        for i in range(break_index, strikes.shape[0]):
            ivs[i] = self.fit(strikes[i])
        prices[break_index:strikes.shape[0]] = Blacks_P(K=strikes[break_index:strikes.shape[0]],
                                                        tenor=self.tenor,
                                                        F=self.F,
                                                        r=self.r,
                                                        iv=ivs[break_index:strikes.shape[0]])


        # for i in range(strikes.shape[0]):
        #     if strikes[i] < self.break_point:
        #         prices[i] = np.exp(-self.r*self.tenor/365)*integrate.quad(self.f_x, self.statr_point, strikes[i], args=(strikes[i],))[0]
        #     else:
        #         ivs = self.fit(strikes[i])
        #         prices[i] = Blacks_P(K=strikes[i], tenor=self.tenor, F=self.F, r=self.r, iv=ivs)
        # prices[0] = 0
        # put_strike = strikes[strikes <= self.S]
        # put_prices = Blacks_P(K=strikes[strikes <= self.S], tenor=self.tenor, F=self.F, r=self.r,
        #                       iv=ivs[strikes <= self.S])
        # call_strike = strikes[strikes >= self.S]
        # call_prices = Blacks_C(K=strikes[strikes >= self.S], tenor=self.tenor, F=self.F, r=self.r,
        #                        iv=ivs[strikes >= self.S])

        V_put = self.comp_V(strikes, prices)
        IV_put = self.comp_IV(strikes, prices)
        # V_call = self.comp_V(call_strike, call_prices)
        # W_put = self.comp_W(put_strike, put_prices)
        # W_call = self.comp_W(call_strike, call_prices)
        # X_put = self.comp_X(put_strike, put_prices)
        # X_call = self.comp_X(call_strike, call_prices)
        # V = V_call + V_put
        # W = W_call + W_put
        # X = X_call + X_put
        RIX = np.exp(self.r*self.tenor/365)*(V_put - IV_put)

        return RIX, V_put, IV_put

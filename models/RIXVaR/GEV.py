# GEV distribution relevant
from scipy.optimize import fsolve
from scipy.optimize import leastsq
from scipy.optimize import fminbound
import numpy as np
class GEV_computer():
    """
    输入3个条件点, 可以输出一个有.cdf.pdf.ppf方法的GEV分布, 注意, ppf方法只对尾部有效, 只在尾部求解可行区域, 至少应该在X0外

    注意,本程序参考:
    Figlewski, S. (2010). Estimating the Implied Risk‐Neutral Density for the US Market Portfolio*.
    In Volatility and Time Series Econometrics (pp. 323–353). Oxford University Press. https://doi.org/10.1093/acprof:oso/9780199549498.003.0015

    该文章只是说可以通过3个方程得到未知数的解, 然而这个方程是非线性方程, 作者并没有证明解的存在性和唯一性
    这里我存疑, 但是确实实际情况都可以得到解, 而且解比较稳定没有明显的证据表明会变动

    """
    def __init__(self, X0, fX0, ALPHA0, X1, fX1, init, shift: int =0):
        """
        计算GEV的参数, 然后生成变量可以给出对应参数下的PDF和CDF, 如果是左尾则给出相应的shift(必须是整数), 如果是右尾则忽略shift
        :param X0:
        :param fX0:
        :param ALPHA0:
        :param X1:
        :param fX1:
        :param init:
        :param shift:
        """
        shift = int(shift)
        self.X0 = X0
        self.fX0 = fX0
        self.ALPHA0 = ALPHA0
        self.X1 = X1
        self.fX1 = fX1
        self.init_guess = init
        if shift:# Meaning left tail Non-zero integer represent True in python
            self.shift = shift
            self.equation = self.equationsL
            self.mu, self.sigma, self.eta = self.solve_para()
            self.cdf = self.GEV_CDFL
            self.pdf = self.GEV_PDFL
            self.ppf = self.GEV_PPFL
        else:
            self.equation = self.equationsR
            self.mu, self.sigma, self.eta = self.solve_para()
            self.cdf = self.GEV_CDFR
            self.pdf = self.GEV_PDFR
            self.ppf = self.GEV_PPFR

    def equationsL(self, p):
        """
        根据3个条件,给出equation用于估计
        :param p: GEV分布的parameter
        :param shift: 因为是估计左尾,需要进行转换# 需要向正轴转换，一般取2*round(F)即可
        :return: equation的结果,期望是(0,0,0)
        """
        mu, sigma, eta = p
        ALPHA0L, X0L, fX0L, X1L, fX1L = self.ALPHA0, self.X0, self.fX0, self.X1, self.fX1
        return (np.exp(-((1 + eta * (((-X0L + self.shift) - mu) / sigma)) ** (-1 / eta))) - (1 - ALPHA0L),
                (1 / sigma) * (((1 + eta * (((-X0L + self.shift) - mu) / sigma)) ** (-1 / eta)) ** (eta + 1)) * np.exp(
                    -(1 + eta * (((-X0L + self.shift) - mu) / sigma)) ** (-1 / eta)) - fX0L,
                (1 / sigma) * (((1 + eta * (((-X1L + self.shift) - mu) / sigma)) ** (-1 / eta)) ** (eta + 1)) *np.exp(
                    -(1 + eta * (((-X1L + self.shift) - mu) / sigma)) ** (-1 / eta)) - fX1L)

    def equationsR(self, p):
        """
        根据3个条件,给出equation用于估计
        :param p: GEV分布的parameter
        :return: equation的结果,期望是(0,0,0)
        """
        mu, sigma, eta = p
        ALPHA0R, X0R, fX0R, X1R, fX1R = self.ALPHA0, self.X0, self.fX0, self.X1, self.fX1
        return (np.exp(-((1+eta*((X0R-mu)/sigma))**(-1/eta)))-ALPHA0R,
                (1/sigma)*(((1+eta*((X0R-mu)/sigma))**(-1/eta))**(eta+1))*np.exp(-(1+eta*((X0R-mu)/sigma))**(-1/eta))-fX0R,
                (1/sigma)*(((1+eta*((X1R-mu)/sigma))**(-1/eta))**(eta+1))*np.exp(-(1+eta*((X1R-mu)/sigma))**(-1/eta))-fX1R)

    def solve_para(self):
        """
        根据self.equation,求解参数,注意,目前的求解方法对于初始值非常敏感,必须给合适的初始值
        :return:
        """
        res = self.equation
        init, n = leastsq(res, self.init_guess)  #先求解个初始值
        mu, sigma, eta =  fsolve(res, init)      #再求个方程解
        return mu, sigma, eta

    def GEV_CDFL(self, X):
        X = -X + self.shift
        return 1 - np.exp(-((1+self.eta*((X-self.mu)/self.sigma))**(-1/self.eta)))

    def GEV_CDFR(self, X):
        return np.exp(-((1 + self.eta * ((X - self.mu) / self.sigma)) ** (-1 / self.eta)))

    def GEV_PDFL(self, X):
        X = -X + self.shift
        return (1/self.sigma)*(((1+self.eta*((X-self.mu)/self.sigma))**(-1/self.eta))**(self.eta+1))*np.exp(-(1+self.eta*((X-self.mu)/self.sigma))**(-1/self.eta))

    def GEV_PDFR(self, X):
        return (1/self.sigma)*(((1+self.eta*((X-self.mu)/self.sigma))**(-1/self.eta))**(self.eta+1))*np.exp(-(1+self.eta*((X-self.mu)/self.sigma))**(-1/self.eta))

    def Diff_L(self, X, alpha):
        return np.fabs(self.cdf(X)-alpha)
    def GEV_PPFL(self, CDF):
        """
        很遗憾,我查阅相关资料几乎没有关于GEV percentile function的资料, 唯一的一个在线资源是给出的ppf结果并不好, 使用sci内置GEV的结果也并不好
        然而查阅图像发现给定参数下PDF和CDF的分布都不错, 故这里仍然采取比较稳健但是更花费计算时间的搜寻方法
        :param CDF:
        :return:
        """
        b = 1
        a = 1-0.02
        count = 0
        while self.Diff_L(a*self.X0, CDF) < self.Diff_L(b*self.X0, CDF):
            a -= 0.02
            b -= 0.02
            if count>50:
                return 0
            else:
                count += 1
        b +=0.02
        return fminbound(self.Diff_L, a*self.X0, b*self.X0, args=(CDF,))

    def Diff_R(self, X, alpha):
        return np.fabs(self.cdf(X)-alpha)
    def GEV_PPFR(self, CDF):
        """

        :param CDF:
        :return:
        """
        a = 1
        b = 1+0.02
        count = 0
        while self.Diff_R(a*self.X0, CDF) > self.Diff_R(b*self.X0, CDF):
            a += 0.02
            b += 0.02
            if count>50:
                return 0
            else:
                count+=1
        a -= 0.02
        return fminbound(self.Diff_R, a*self.X0, b*self.X0, args=(CDF,))



# #GitHub上某个项目的这个部分的分布函数，感觉是写错了,但是不确定是不是GEV的一种改进
# def equations(p, *con):
#     mu, sigma, eta = p
#     ALPHA0, X0, fX0, X1, fX1 = con
#     return (np.exp(-(1+eta*((X0-mu)/sigma))**(-1/eta))-ALPHA0,
#             (np.exp(1+eta)**((-1/eta)-1))*np.exp(-(1+eta*((X0-mu)/sigma))**(-1/eta))-fX0,
#            (np.exp(1+eta)**((-1/eta)-1))*np.exp(-(1+eta*((X1-mu)/sigma))**(-1/eta))-fX1)
# def GEV(st, mu, sigma, eta):
#     return (np.exp(1+eta)**((-1/eta)-1))*np.exp(-(1+eta*((st-mu)/sigma))**(-1/eta))


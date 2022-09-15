"""
实现插值函数的估计
"""


from statsmodels.nonparametric import kernel_regression
import numpy as np
class LLkernel_L(object):
    """
    实现一个LLkernelregression_linear的插值函数
    使用训练集创造函数endog,exog训练集,eps求导的delta
    用.fit(K)方法,返回插值结果
    """

    def __init__(self, endog, exog, eps=0.01):
        self.exog = np.asarray(exog)
        self.endog = np.asarray(endog)
        self.K_right = self.exog.max()
        self.K_left = self.exog.min()
        self.iv_right = self.endog[exog == self.K_right][0]
        self.iv_left = self.endog[exog == self.K_left][0]

        self.model = kernel_regression.KernelReg(endog, exog, 'c')

        self.eps = eps
        self.slop_left = min(0, self.comp_slop(self.K_left))
        self.slop_right = max(0, self.comp_slop(self.K_right))

    def inner_fit(self, k):
        """
        注意, statsmodel的kernel regression返回tuple of ndarray,
        第一个位置是我们需要的, 第二个位置是置信度

        :param k:
        :return:-> numpy.float64
        """
        return self.model.fit(np.array([k]))[0][0]

    def comp_slop(self, k):
        slop = (self.inner_fit(k + self.eps) - self.inner_fit(k - self.eps)) / (2 * self.eps)
        return slop

    def fit(self, k):
        """

        :param k:
        :return:-> numpy.float64
        """

        if k < self.K_left:
            return self.slop_left * (k - self.K_left) + self.iv_left
        elif self.K_left <= k <= self.K_right:
            return self.inner_fit(k)
        else:
            return self.slop_right * (k - self.K_right) + self.iv_right

class LLkernel_H(object):
    """
     实现一个LLkernelregression_horizontal的插值函数
     使用训练集创造函数endog,exog训练集,eps求导的delta
     用.fit(K)方法,返回插值结果

     进行了一下修正, 现在边缘部分返回的是正常的插值结果, 而不是之前的最大值最小值
     """

    def __init__(self, endog, exog):
        self.exog = np.asarray(exog)
        self.endog = np.asarray(endog)
        self.model = kernel_regression.KernelReg(endog, exog, 'c')

        self.K_right = self.exog.max()
        self.K_left = self.exog.min()
        self.iv_right = self.inner_fit(self.K_right)
        self.iv_left = self.inner_fit(self.K_left)

    def inner_fit(self, k):
        """
        注意, statsmodel的kernel regression返回tuple of ndarray,
        第一个位置是我们需要的, 第二个位置是置信度

        :param k:
        :return:-> numpy.float64
        """
        return self.model.fit(np.array([k]))[0][0]

    def fit(self, k):
        """

        :param k:
        :return:-> numpy.float64
        """

        if k < self.K_left:
            return self.iv_left
        elif self.K_left <= k <= self.K_right:
            return self.inner_fit(k)
        else:
            return self.iv_right

class LLkernel_H_j(object):
    """
    实现一个LLkernelregression_horizontal的插值函数
    使用训练集创造函数endog,exog训练集,eps求导的delta
    用.fit(K)方法,返回插值结果
    """
    def __init__(self, endog, exog):
        self.exog = np.asarray(exog)
        self.endog = np.asarray(endog)
        self.model = kernel_regression.KernelReg(endog, exog, 'c')

        self.K_right = self.exog.max()
        self.K_left = self.exog.min()
        self.iv_right = self.endog[exog == self.K_right][0]
        self.iv_left = self.endog[exog == self.K_left][0]

    def inner_fit(self, k) :
        """
        注意, statsmodel的kernel regression返回tuple of ndarray,
        第一个位置是我们需要的, 第二个位置是置信度

        :param k:
        :return:-> numpy.float64
        """
        return self.model.fit(np.array([k]))[0][0]

    def fit(self, k) :
        """

        :param k:
        :return:-> numpy.float64
        """

        if k < self.K_left:
            return self.iv_left
        elif self.K_left <= k <= self.K_right:
            return self.inner_fit(k)
        else:
            return self.iv_right


#%%

import statsmodels.api as sm
from patsy import dmatrix
#reference:https://www.analyticsvidhya.com/blog/2018/03/introduction-regression-splines-python-codes/

class Spline_H(object):

    def __init__(self, endog, exog, knot, order=4, intercept=False, OLS=True):
        self.exog = np.asarray(exog)
        self.endog = np.asarray(endog)
        self.knot = knot
        self.order = order
        self.intercept = intercept
        # no evidence shows that when k==0 iv==0, so include intercept
        # degree of freedom is 1+order+K, 6 for a 4 order 1 knot spline, so at least 3 call and 3 put option
        matrix = dmatrix("bs(self.exog, knots=(self.knot,), degree=self.order, include_intercept=self.intercept)",
                         {"self.exog": self.exog},
                         return_type='dataframe')
        self.model = sm.GLM(self.endog, matrix).fit() if OLS else sm.RLM(self.endog, matrix).fit()

        self.K_left = self.exog.min()
        self.K_right = self.exog.max()
        self.iv_left = self.inner_fit(self.K_left)
        self.iv_right = self.inner_fit(self.K_right)

    def inner_fit(self, K):
        sample = np.array([self.K_left, K, self.K_right])
        iv = self.model.predict(dmatrix("bs(sample, knots=(self.knot,), degree=self.order, include_intercept=self.intercept)",
                                {"sample": sample},
                                return_type='dataframe'))[1]
        return iv

    def fit(self, K):

        if K <= self.K_left:
            return self.iv_left
        elif self.K_left < K < self.K_right:
            return self.inner_fit(K)
        else:
            return self.iv_right


class Spline_L(object):

    def __init__(self, endog, exog, knot, eps=0.01, order=4, intercept=False, OLS=True):
        self.exog = np.asarray(exog)
        self.endog = np.asarray(endog)
        self.knot = knot
        self.order = order
        self.intercept = intercept
        # no evidence shows that when k==0 iv==0, so include intercept
        # degree of freedom is 1+order+K, 6 for a 4 order 1 knot spline, so at least 3 call and 3 put option
        matrix = dmatrix("bs(self.exog, knots=(self.knot,), degree=self.order, include_intercept=self.intercept)",
                         {"self.exog": self.exog},
                         return_type='dataframe')
        self.model = sm.GLM(self.endog, matrix).fit() if OLS else sm.RLM(self.endog, matrix).fit()

        self.K_left = self.exog.min()
        self.K_right = self.exog.max()

        self.eps = eps
        self.slop_left = min(0, self.comp_slop(self.K_left))
        self.slop_right = max(0, self.comp_slop(self.K_right))

    def inner_fit(self, K):
        sample = np.array([self.K_left, K, self.K_right])
        iv = self.model.predict(dmatrix("bs(sample, knots=(self.knot,), degree=self.order, include_intercept=self.intercept)",
                                {"sample": sample},
                                return_type='dataframe'))[1]
        return iv

    def comp_slop(self, k):
        slop = (self.inner_fit(k + self.eps) - self.inner_fit(k - self.eps)) / (2 * self.eps)
        return slop

    def fit(self, k):
        if k < self.K_left:
            return self.slop_left * (k - self.K_left) + self.iv_left
        elif self.K_left <= k <= self.K_right:
            return self.inner_fit(k)
        else:
            return self.slop_right * (k - self.K_right) + self.iv_right
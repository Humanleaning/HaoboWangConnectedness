"""
实际使用时请开启try-except模块
"""

from models.RIXVaR.PolationTechnics import LLkernel_L
from models.RIXVaR.PolationTechnics import LLkernel_H
from models.RIXVaR.PolationTechnics import Spline_H
from models.RIXVaR.RiskNeutralDensityComputer import VaRComputer_p
from models.RIXVaR.RiskNeutralDensityComputer import MovementComputer
from models.RIXVaR.GEV import GEV_computer

def compute_movments_var_kernel(info_ex, date, VaRandVol, alphas=(0.01, 0.05, 0.1)):
    # try:
    info = info_ex.getoneday(date)
    # except:
    #     for alpha in alphas:
    #         VaRandVol.loc[date, ('VaR_30_' + str(alpha))] = 0.1
    #     return VaRandVol

    # try:
    if info['insert']:
        S = info['S']
        F = info['F']
        r = info['r']
        tenor_front = info['tenor_front']
        tenor_rear = info['tenor_rear']
        VaRandVol.loc[date, 'S'] = S
        #### movement ####
        chine_front = info['chine_front']
        chine_rear = info['chine_rear']
        krh1 = LLkernel_H(chine_front.impl_volatility, chine_front.strike_price)
        krh2 = LLkernel_H(chine_rear.impl_volatility, chine_rear.strike_price)
        mc = MovementComputer(S, F, r, insert=True, args=(krh1, krh2, tenor_front, tenor_rear))
        VaRandVol.loc[date, 'vol'] = mc.vol
        VaRandVol.loc[date, 'skew'] = mc.skew
        VaRandVol.loc[date, 'kurto'] = mc.kurto
        #### VaR ####
        chine_front_put = info['chine_front_put']
        chine_rear_put = info['chine_rear_put']
        kr1 = LLkernel_L(chine_front_put.impl_volatility, chine_front_put.strike_price)
        kr2 = LLkernel_L(chine_rear_put.impl_volatility, chine_rear_put.strike_price)
        vc = VaRComputer_p(S, F, r, insert=True, args=(kr1, kr2, tenor_front, tenor_rear))

        ##选取断点###
        ##方式一, 选取带有线性插值的部分##
        # if kr1.K_left != kr2.K_left:
        #     X0L = max(kr1.K_left, kr2.K_left)
        #     X1L = min(kr1.K_left, kr2.K_left)
        #     ignore0, ALPHA0L, fX0L = vc.comp_rn(X0L)
        #     ignore1, ALPHA1L, fX1L = vc.comp_rn(X1L)
        # else:
        #     X1L = kr1.K_left
        #     ignore1, ALPHA1L, fX1L = vc.comp_rn(X1L)
        #     ALPHA0L = ALPHA1L+0.03
        #     X0L = vc.comp_VaR(ALPHA0L)
        #     ignore0, ALPHA0L, fX0L = vc.comp_rn(X0L)
        ##方式三, 最大置信区间中选取
        ##方式三, 在0.05, 最大置信度区间边界中选取最小的那个##
        X1L = max(kr1.K_left, kr2.K_left) + kr1.eps
        ignore1, ALPHA1L, fX1L = vc.comp_rn(X1L)
        if ALPHA1L >= 0.05:
            pass
        else:
            ALPHA1L = 0.05
            X1L = vc.comp_VaR(ALPHA1L)
            ignore1, ALPHA1L, fX1L = vc.comp_rn(X1L)
        ALPHA0L = ALPHA1L + 0.03
        X0L = vc.comp_VaR(ALPHA0L)
        ignore0, ALPHA0L, fX0L = vc.comp_rn(X0L)

        gev = GEV_computer(X0L, fX0L, ALPHA0L, X1L, fX1L, (F, mc.vol * F, -0.25), shift=2 * round(F))
        for alpha in alphas:
            if alpha < ALPHA0L:
                VaRandVol.loc[date, ('VaR_30_' + str(alpha))] = gev.ppf(alpha)
                VaRandVol.loc[date, 'miu'] = gev.mu
                VaRandVol.loc[date, 'sigma'] = gev.sigma
                VaRandVol.loc[date, 'eta'] = gev.eta
            else:
                VaRandVol.loc[date, ('VaR_30_' + str(alpha))] = vc.comp_VaR(alpha)
    else:  # 进入直接有一个30天的模式
        S = info['S']
        F = info['F']
        r = info['r']
        tenor = info['tenor']
        VaRandVol.loc[date, 'S'] = S
        # ----------vol
        chine = info['chine']
        krh = LLkernel_H(chine.impl_volatility, chine.strike_price)
        mc = MovementComputer(S, F, r, insert=False, args=krh)
        VaRandVol.loc[date, 'vol'] = mc.vol
        VaRandVol.loc[date, 'skew'] = mc.skew
        VaRandVol.loc[date, 'kurto'] = mc.kurto

        # -----------VaR
        chine_put = info['chine_put']
        kr = LLkernel_L(chine_put.impl_volatility, chine_put.strike_price)
        vc = VaRComputer_p(S, F, r, tenor=tenor, insert=False, args=kr)

        ###方式三####
        X1L = kr.K_left + kr.eps
        ignore1, ALPHA1L, fX1L = vc.comp_rn(X1L)
        if ALPHA1L >= 0.05:
            pass
        else:
            ALPHA1L = 0.05
            X1L = vc.comp_VaR(ALPHA1L)
            ignore1, ALPHA1L, fX1L = vc.comp_rn(X1L)
        ALPHA0L = ALPHA1L + 0.03
        X0L = vc.comp_VaR(ALPHA0L)
        ignore0, ALPHA0L, fX0L = vc.comp_rn(X0L)

        gev = GEV_computer(X0L, fX0L, ALPHA0L, X1L, fX1L, (F, mc.vol * F, -0.25), shift=2 * round(F))
        for alpha in alphas:
            if alpha < ALPHA0L:
                VaRandVol.loc[date, ('VaR_30_' + str(alpha))] = gev.ppf(alpha)
                VaRandVol.loc[date, 'miu'] = gev.mu
                VaRandVol.loc[date, 'sigma'] = gev.sigma
                VaRandVol.loc[date, 'eta'] = gev.eta
            else:
                VaRandVol.loc[date, ('VaR_30_' + str(alpha))] = vc.comp_VaR(alpha)
    return VaRandVol

    # except:
    #     for alpha in alphas:
    #         VaRandVol.loc[date, ('VaR_30_' + str(alpha))] = 0
    #     return VaRandVol

def compute_movments_var_spline(info_ex, date, VaRandVol, alphas=(0.01, 0.05, 0.1)):
    # try:
    info = info_ex.getoneday(date)
    # except:
    #     for alpha in alphas:
    #         VaRandVol.loc[date, ('VaR_30_' + str(alpha))] = 0.1
    #     return VaRandVol

    # try:
    if info['insert']:
        S = info['S']
        F = info['F']
        r = info['r']
        tenor_front = info['tenor_front']
        tenor_rear = info['tenor_rear']
        VaRandVol.loc[date, 'S'] = S

        #### calculate risk neutral movemets ####
        chine_front = info['chine_front']
        chine_rear = info['chine_rear']
        pol_model1 = Spline_H(chine_front.impl_volatility, chine_front.strike_price, info['F_front'])
        pol_model2 = Spline_H(chine_rear.impl_volatility, chine_rear.strike_price, info['F_rear'])
        mc = MovementComputer(S, F, r, insert=True, args=(pol_model1, pol_model2, tenor_front, tenor_rear))
        VaRandVol.loc[date, 'vol'] = mc.vol
        VaRandVol.loc[date, 'skew'] = mc.skew
        VaRandVol.loc[date, 'kurto'] = mc.kurto
        #### calculate risk neutral quantile ####
        vc = VaRComputer_p(S, F, r, insert=True, args=(pol_model1, pol_model2, tenor_front, tenor_rear))
        ##方式一, 选取带有线性插值的部分##
        # if kr1.K_left != kr2.K_left:
        #     X0L = max(kr1.K_left, kr2.K_left)
        #     X1L = min(kr1.K_left, kr2.K_left)
        #     ignore0, ALPHA0L, fX0L = vc.comp_rn(X0L)
        #     ignore1, ALPHA1L, fX1L = vc.comp_rn(X1L)
        # else:
        #     X1L = kr1.K_left
        #     ignore1, ALPHA1L, fX1L = vc.comp_rn(X1L)
        #     ALPHA0L = ALPHA1L+0.03
        #     X0L = vc.comp_VaR(ALPHA0L)
        #     ignore0, ALPHA0L, fX0L = vc.comp_rn(X0L)
        ##方式三, 最大置信区间中选取
        ##方式三, 在0.05, 最大置信度区间边界中选取最小的那个##
        X1L = max(pol_model1.K_left, pol_model2.K_left) + vc.eps
        ignore1, ALPHA1L, fX1L = vc.comp_rn(X1L)
        if ALPHA1L >= 0.05:
            pass
        else:
            ALPHA1L = 0.05
            X1L = vc.comp_VaR(ALPHA1L)
            ignore1, ALPHA1L, fX1L = vc.comp_rn(X1L)
        ALPHA0L = ALPHA1L + 0.03
        X0L = vc.comp_VaR(ALPHA0L)
        ignore0, ALPHA0L, fX0L = vc.comp_rn(X0L)

        gev = GEV_computer(X0L, fX0L, ALPHA0L, X1L, fX1L, (F, mc.vol * F, -0.25), shift=2 * round(F))
        for alpha in alphas:
            if alpha < ALPHA0L:
                VaRandVol.loc[date, ('VaR_30_' + str(alpha))] = gev.ppf(alpha)
                VaRandVol.loc[date, 'miu'] = gev.mu
                VaRandVol.loc[date, 'sigma'] = gev.sigma
                VaRandVol.loc[date, 'eta'] = gev.eta
            else:
                VaRandVol.loc[date, ('VaR_30_' + str(alpha))] = vc.comp_VaR(alpha)
    else:  # 进入直接有一个30天的模式
        S = info['S']
        F = info['F']
        r = info['r']
        tenor = info['tenor']
        VaRandVol.loc[date, 'S'] = S
        # ----------vol
        chine = info['chine']
        pol_model = Spline_H(chine.impl_volatility, chine.strike_price, F)
        mc = MovementComputer(S, F, r, insert=False, args=pol_model)
        VaRandVol.loc[date, 'vol'] = mc.vol
        VaRandVol.loc[date, 'skew'] = mc.skew
        VaRandVol.loc[date, 'kurto'] = mc.kurto
        # -----------VaR
        vc = VaRComputer_p(S, F, r, insert=False, args=pol_model)

        ###方式三####
        X1L = pol_model.K_left + vc.eps
        ignore1, ALPHA1L, fX1L = vc.comp_rn(X1L)
        if ALPHA1L >= 0.05:
            pass
        else:
            ALPHA1L = 0.05
            X1L = vc.comp_VaR(ALPHA1L)
            ignore1, ALPHA1L, fX1L = vc.comp_rn(X1L)
        ALPHA0L = ALPHA1L + 0.03
        X0L = vc.comp_VaR(ALPHA0L)
        ignore0, ALPHA0L, fX0L = vc.comp_rn(X0L)

        gev = GEV_computer(X0L, fX0L, ALPHA0L, X1L, fX1L, (F, mc.vol * F, -0.25), shift=2 * round(F))
        for alpha in alphas:
            if alpha < ALPHA0L:
                VaRandVol.loc[date, ('VaR_30_' + str(alpha))] = gev.ppf(alpha)
                VaRandVol.loc[date, 'miu'] = gev.mu
                VaRandVol.loc[date, 'sigma'] = gev.sigma
                VaRandVol.loc[date, 'eta'] = gev.eta
            else:
                VaRandVol.loc[date, ('VaR_30_' + str(alpha))] = vc.comp_VaR(alpha)
    return VaRandVol
    # except:
    #     for alpha in alphas:
    #         VaRandVol.loc[date, ('VaR_30_' + str(alpha))] = 0
    #     return VaRandVol



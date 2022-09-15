from models.RIXVaR.PolationTechnics import LLkernel_L
from models.RIXVaR.PolationTechnics import Spline_H
from models.RIXVaR.RiskNeutralDensityComputer import VaRComputer_p
from models.RIXVaR.GEV import GEV_computer
from models.RIXVaR.GEV_style_RIX import GEV_RIX_Compute

# read the innitial data and compute gev & 2polation function （if we only consider downside risk）
# Finally， what we actually need is 4000 points of（strike_price, put_price)
# This can be calculated using a if else: if outside x0, use gev pdf*(k-x) intigrate (0, k) else inside x0, using volatility





def compute_rix_spline(info_ex, date, RIXandVol, precision=2000):
    try:
        info = info_ex.getoneday(date)
        (mu, sigma, eta) = RIXandVol.loc[date, ['miu', 'sigma', 'eta']].values
        if eta > 0:
            return RIXandVol
    except:
        RIXandVol.loc[date, 'RIX'] = -0.001
        return RIXandVol



    try:
        if info['insert']:
            S = info['S']
            F = info['F']
            r = info['r']
            tenor_front = info['tenor_front']
            tenor_rear = info['tenor_rear']
            RIXandVol.loc[date, 'S'] = S

            #### calculate risk neutral movemets ####
            chine_front = info['chine_front']
            chine_rear = info['chine_rear']
            pol_model1 = Spline_H(chine_front.impl_volatility, chine_front.strike_price, info['F_front'])
            pol_model2 = Spline_H(chine_rear.impl_volatility, chine_rear.strike_price, info['F_rear'])
            # mc = MovementComputer(S, F, r, insert=True, args=(pol_model1, pol_model2, tenor_front, tenor_rear))
            # RIXandVol.loc[date, 'vol'] = mc.vol
            # RIXandVol.loc[date, 'skew'] = mc.skew
            # RIXandVol.loc[date, 'kurto'] = mc.kurto

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

            gev = GEV_computer(X0L, fX0L, ALPHA0L, X1L, fX1L, (mu, sigma, eta), shift=2 * round(F))

            gev_indicator = GEV_RIX_Compute(gev, X0L, S, F, r, insert=True, args=(pol_model1, pol_model2, tenor_front, tenor_rear),precision=precision)
            RIX = gev_indicator.RIX_L
            RIXandVol.loc[date, 'RIX'] = RIX
            # for alpha in alphas:
            #     if alpha < ALPHA0L:
            #         RIXandVol.loc[date, ('VaR_30_' + str(alpha))] = gev.ppf(alpha)
            #         RIXandVol.loc[date, 'miu'] = gev.mu
            #         RIXandVol.loc[date, 'sigma'] = gev.sigma
            #         RIXandVol.loc[date, 'eta'] = gev.eta
            #     else:
            #         RIXandVol.loc[date, ('VaR_30_' + str(alpha))] = vc.comp_VaR(alpha)
        else:  # 进入直接有一个30天的模式
            S = info['S']
            F = info['F']
            r = info['r']
            tenor = info['tenor']
            RIXandVol.loc[date, 'S'] = S
            # ----------vol
            chine = info['chine']
            pol_model = Spline_H(chine.impl_volatility, chine.strike_price, F)
            # mc = MovementComputer(S, F, r, insert=False, args=pol_model)
            # RIXandVol.loc[date, 'vol'] = mc.vol
            # RIXandVol.loc[date, 'skew'] = mc.skew
            # RIXandVol.loc[date, 'kurto'] = mc.kurto
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

            gev = GEV_computer(X0L, fX0L, ALPHA0L, X1L, fX1L, (mu, sigma, eta), shift=2 * round(F))
            gev_indicator = GEV_RIX_Compute(gev, X0L, S, F, r, insert=False, args=pol_model, precision=precision)
            RIX = gev_indicator.RIX_L
            RIXandVol.loc[date, 'RIX'] = RIX
            # for alpha in alphas:
            #     if alpha < ALPHA0L:
            #         RIXandVol.loc[date, ('VaR_30_' + str(alpha))] = gev.ppf(alpha)
            #         RIXandVol.loc[date, 'miu'] = gev.mu
            #         RIXandVol.loc[date, 'sigma'] = gev.sigma
            #         RIXandVol.loc[date, 'eta'] = gev.eta
            #     else:
            #         RIXandVol.loc[date, ('VaR_30_' + str(alpha))] = vc.comp_VaR(alpha)
        return RIXandVol
    except:
        RIXandVol.loc[date, 'RIX'] = 0
        return RIXandVol


def compute_rix_kernel(info_ex, date, RIXandVol, precision=2000):
    try:
        info = info_ex.getoneday(date)
        (mu, sigma, eta) = RIXandVol.loc[date, ['miu', 'sigma', 'eta']].values
        if eta > 0:
            return RIXandVol
    except:
        RIXandVol.loc[date, 'RIX'] = -0.001
        return RIXandVol



    try:
        if info['insert']:
            S = info['S']
            F = info['F']
            r = info['r']
            tenor_front = info['tenor_front']
            tenor_rear = info['tenor_rear']
            RIXandVol.loc[date, 'S'] = S
            #### movement ####
            # chine_front = info['chine_front']
            # chine_rear = info['chine_rear']
            # krh1 = LLkernel_H(chine_front.impl_volatility, chine_front.strike_price)
            # krh2 = LLkernel_H(chine_rear.impl_volatility, chine_rear.strike_price)
            # mc = MovementComputer(S, F, r, insert=True, args=(krh1, krh2, tenor_front, tenor_rear))
            # RIXandVol.loc[date, 'vol'] = mc.vol
            # RIXandVol.loc[date, 'skew'] = mc.skew
            # RIXandVol.loc[date, 'kurto'] = mc.kurto
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

            gev = GEV_computer(X0L, fX0L, ALPHA0L, X1L, fX1L, (mu, sigma, eta), shift=2 * round(F))
            gev_indicator = GEV_RIX_Compute(gev, X0L, S, F, r, insert=True, args=(kr1, kr2, tenor_front, tenor_rear), precision=precision)
            RIX = gev_indicator.RIX_L
            RIXandVol.loc[date, 'RIX'] = RIX

            # for alpha in alphas:
            #     if alpha < ALPHA0L:
            #         RIXandVol.loc[date, ('VaR_30_' + str(alpha))] = gev.ppf(alpha)
            #         RIXandVol.loc[date, 'miu'] = gev.mu
            #         RIXandVol.loc[date, 'sigma'] = gev.sigma
            #         RIXandVol.loc[date, 'eta'] = gev.eta
            #     else:
            #         RIXandVol.loc[date, ('VaR_30_' + str(alpha))] = vc.comp_VaR(alpha)
        else:  # 进入直接有一个30天的模式
            S = info['S']
            F = info['F']
            r = info['r']
            tenor = info['tenor']
            RIXandVol.loc[date, 'S'] = S
            # ----------vol
            # chine = info['chine']
            # krh = LLkernel_H(chine.impl_volatility, chine.strike_price)
            # mc = MovementComputer(S, F, r, insert=False, args=krh)
            # RIXandVol.loc[date, 'vol'] = mc.vol
            # RIXandVol.loc[date, 'skew'] = mc.skew
            # RIXandVol.loc[date, 'kurto'] = mc.kurto

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

            gev = GEV_computer(X0L, fX0L, ALPHA0L, X1L, fX1L, (mu, sigma, eta), shift=2 * round(F))
            gev_indicator = GEV_RIX_Compute(gev, X0L, S, F, r, insert=False, args=kr, precision=precision)
            RIX = gev_indicator.RIX_L
            RIXandVol.loc[date, 'RIX'] = RIX

            # for alpha in alphas:
            #     if alpha < ALPHA0L:
            #         RIXandVol.loc[date, ('VaR_30_' + str(alpha))] = gev.ppf(alpha)
            #         RIXandVol.loc[date, 'miu'] = gev.mu
            #         RIXandVol.loc[date, 'sigma'] = gev.sigma
            #         RIXandVol.loc[date, 'eta'] = gev.eta
            #     else:
            #         RIXandVol.loc[date, ('VaR_30_' + str(alpha))] = vc.comp_VaR(alpha)
        return RIXandVol
    except:
        RIXandVol.loc[date, 'RIX'] = 0
        return RIXandVol
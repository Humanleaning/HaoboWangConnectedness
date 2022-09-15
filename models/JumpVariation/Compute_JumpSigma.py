




from models.JumpVariation.SigmaCalculator import SigmaandTheta
from models.JumpVariation.Interpolation import InterpolationData, Convert2price
from models.JumpVariation.JumpCalculator import LV_Tao
import copy


def compute_JS(info_ex, date, df):
    # try:
    info = info_ex.getoneday(date)
    # except:
    #     for alpha in alphas:
    #         VaRandVol.loc[date, ('VaR_30_' + str(alpha))] = 0.1
    #     return VaRandVol

    # try:
    info = InterpolationData(info)
    info = Convert2price(info)
    info_Sigma = copy.deepcopy(info)
    sigma, theta = SigmaandTheta(info_Sigma)
    alpha, phi, LV = LV_Tao(theta, info)
    # except:
    #     sigma = -0.01
    #     alpha = -0.01
    #     phi = -0.01
    #     LV = -0.01

    df.loc[date, "sigma"] = sigma
    df.loc[date, 'alpha'] = alpha
    df.loc[date, 'phi'] = phi
    df.loc[date, 'LV'] = LV

    return df
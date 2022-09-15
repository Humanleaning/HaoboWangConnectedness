from numba import njit
import numpy as np

#f函数的原函数、一二阶导数
##@njit
def Letter_f(x, y, k):
    u = np.complex(x,y)
    return (np.square(u) - u )*np.exp((u-1)*k)

##@njit
def Flower_Lhat_tT(x, y, option_prices, log_strikes, log_x):
    """
    return numpy.complex64
    """
    return (1 + np.exp(-log_x) * np.sum(Letter_f(x, y, (log_strikes[:-1] - log_x)) *
                                  option_prices[:-1] *
                                  (log_strikes[1:]-log_strikes[:-1])))

##@njit
def Flower_Lhat_tTao(x, y, optiondata, forwarddata):
    l = 1+0j
    for tenor_index, F in enumerate(forwarddata):
        l *= Flower_Lhat_tT(x, y, option_prices=optiondata[:,0, tenor_index],
                       log_strikes=optiondata[:,1, tenor_index],
                       log_x=F)
    return l

#Find uhat and calculate sigma theta
##@njit
def find_uhat(BSIV, tenors, optiondata, forwarddata):
    k = len(forwarddata)

    ubar = np.sqrt(2 / np.sum(tenors) * np.log(10) / np.square(BSIV)) #if (len(tenors)>1) else np.sqrt(2 / tenors * np.log(10) / np.square(BSIV))
    u_grid = np.arange(0.1, ubar, 0.1)
    L = np.empty(u_grid.shape, dtype=np.complex64)
    for i, ui in enumerate(u_grid):
      L[i] = Flower_Lhat_tTao(0, ui, optiondata, forwarddata)
    # uhat1 based on abs(L(u)): 1st time abs(L(u)) <= 0.2
    uhat1_ind = np.where(np.abs(L) <= np.power(0.1, k))[0]
    if uhat1_ind.size > 0:
      uhat1 = u_grid[uhat1_ind[0]]
    else:
      uhat1 = ubar
    # uhat2 -- abs(L(u)) attains minimum on [0,ubar]
    uhat2 = u_grid[(np.abs(L) == np.min(np.abs(L)))][0]
    # uhat is the minimum of uhat1 and uhat2
    uhat = min(uhat1, uhat2)
    return uhat

##@njit
def CalSigmaTheta(uhat, tenors, optiondata, forwarddata, multiplier=7, tenor=30):

    sigma = np.sqrt(-2/(np.sum(tenors))/np.square(uhat)*np.log(np.abs(Flower_Lhat_tTao(0, uhat, optiondata, forwarddata))))
    theta = multiplier*sigma*np.sqrt(tenor/252)
    return sigma, theta


def SigmaandTheta(datalist):

    """
    把所有的初始信息都转化为(x,y)组合，然后输入给

    :param theta:
    :return:
    """
    Option_list = []
    log_forward_list = []
    tenor_list = []
    BSIV_list = []
    for sublist in datalist:
        BSIV, tenor, interest, spot, forward, data = sublist
        tenor = tenor/365
        BSIV_list.append(BSIV)
        log_forward_list.append(np.log(forward))
        tenor_list.append(tenor)
        data[:, 1] = np.log(data[:, 1])
        Option_list.append(data)



    BSIV = BSIV_list[0]
    if len(log_forward_list) > 1:
        optiondata  = np.dstack(Option_list)
        log_forwarddata = np.hstack(log_forward_list)
        tenors      = np.hstack(tenor_list)
    else:
        optiondata  =np.dstack((Option_list[0], np.zeros(Option_list[0].shape)))
        log_forwarddata = log_forward_list[0]
        tenors      = tenor_list[0]


    uhat = find_uhat(BSIV, tenors, optiondata, log_forwarddata)
    sigma,theta = CalSigmaTheta(uhat, tenors, optiondata, log_forwarddata, multiplier=7, tenor=5)
    return sigma, theta



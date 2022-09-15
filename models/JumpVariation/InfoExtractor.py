"""
目标功能，从原始数据中提取能够用于计算JumpVariation的数据

# 输入所有需要的数据集, 然后进行数据清洗
# getoneday方法, 输入日期, 获取对应数据集的日信息, 即市场信息:
#   1）所有在8-42以内的期权链(option_price, strike_price, impl_volatility), [np.exp(np.log(S)-8*BSIV*np.sqrt(T)) ,np.exp(np.log(S)+8*BSIV*np.sqrt(T))
#   2）对应的r,q,F,T
"""



import numpy as np
def Get_r(interestset, date, tenor):
    Interest = interestset.loc[date]
    # if np.all(np.diff(Interest.tenor) > 0):
    return np.interp(tenor, Interest.tenor.values, Interest.interest.values)


# %%
# 用于提取IvyDB US的数据
class IvyDBUS_InfoExtractor(object):

    def __init__(self, Options, Forward, Interest, Spot):
        """
        -------------------------数据筛选----------------------
        进行的筛选有

        ###类型筛选
        0)(在计算VaR的时候)只使用看跌期权或看涨期权

        ###价格筛选
        2)数据筛选, 去除bid_ask spread过大的期权
        3)数据筛选, 相同日期\tenor\type, 去掉midprice相同的期权
        4)数据筛选, 去除best_bid过小的期权
        6)数据筛选, 按照VIX官方进行筛选

        ###需要市场信息的筛选
        1)数据筛选, 只留下OTM期权
        5)数据筛选, 去掉vega过小的期权 (不进行)

        ###数量筛选
        OTM期权的数量必须大于mink


        :param Options:
        :param Forward:
        :param Interest:
        :param Spot:
        :param tenor: 默认是30
        :param option_type:
        """

        # 只留下tenor在[8-42]的期权
        Options = Options.loc[(Options.tenor >=8) & (Options.tenor <=45)]
        Options = Options.loc[Options.impl_volatility > 0]



        # 1)去掉askbidspread过大的
        Options['ab_spread'] = (Options.best_offer - Options.best_bid)
        Options = Options.loc[Options.ab_spread < 5 * Options.best_bid]
        # 2)去掉midprice相同的, 保留距离中值最小的那个
        Options['mid_price'] = (Options.best_bid + Options.best_offer) / 2
        Options['distance_to_ATM'] = np.fabs(Options.moneynessS - 1)
        Options.reset_index(inplace=True)
        Options = Options.sort_values(['date', 'cp_flag', 'tenor', 'distance_to_ATM'], ascending=True)
        Options.drop_duplicates(subset=['date', 'tenor', 'cp_flag', 'mid_price'], keep='first', inplace=True, ignore_index=True)  # 保留距离中值最小的那个
        Options.set_index('date', inplace=True)
        # 3)取消best bid过小的
        Options = Options.loc[Options.best_bid >= 0]



        OTM_Options = Options.loc[((Options.cp_flag == 'C') & (Options.moneynessF >= 1)) |
                                  ((Options.cp_flag == 'P') & (Options.moneynessF <= 1))]
        OTM_Options.reset_index(inplace=True)
        OTM_Options = OTM_Options.sort_values(['date', 'tenor', 'strike_price'], ascending=True)
        OTM_Options.set_index('date', inplace=True)
        self.OTM_Options = OTM_Options
        self.Forward = Forward
        self.Interest = Interest
        self.Spot = Spot


    def getoneday(self, date):

        OTM_Option = self.OTM_Options.loc[date]
        out_put_list = []
        tenors = OTM_Option.tenor.unique()
        for tenor in tenors:
            options = OTM_Option.loc[OTM_Option.tenor == tenor]
            r = Get_r(self.Interest, date, tenor)
            options.interest = r
            interest = options['interest'].unique()
            spot = options['spot'].unique()
            forward = options['forward_price'].unique()
            data = options[['impl_volatility', 'strike_price']].values



            out_put_list.append([tenor,interest,spot,forward,data])

        return out_put_list

#%%
# 用于提取IvyDB European的数据
class IvyDBEurope_InfoExtractor(object):

    def __init__(self, market, Options, Forwards, Dividend, Interests, Spots):
        """

        :param market: str 要计算oi的市场
        :param Options:
        :param Forwards:
        :param Interests:
        :param Spots:

        """

        # 只留下tenor在[8-42]的期权
        Options = Options.loc[(Options.tenor >= 8) & (Options.tenor <= 45)]
        # 去除iv小于0期权
        Options = Options.loc[Options.otheriv>0]

        minbids = {'AEX': 0.1, 'CAC 40': 0.1, 'DAX': 0.1, 'FTSE MIB': 0.1, 'FTSE 100': 0.1}
        minbid = minbids[market]
        multipliers = {'AEX': 10, 'CAC 40': 10, 'DAX': 10, 'FTSE MIB': 10, 'FTSE 100': 10}
        multiplier = multipliers[market]


        # 去掉best_bid过小的
        Options = Options.loc[Options.best_bid >= minbid]
        # 去掉midprice相同的
        Options['mid_price'] = (Options.best_bid + Options.best_offer) / 2
        Options['distance_to_ATM'] = np.fabs(Options.moneynessS - 1)
        Options.reset_index(inplace=True)
        Options = Options.sort_values(['date', 'cp_flag', 'tenor', 'distance_to_ATM'], ascending=True)
        Options.drop_duplicates(subset=['date', 'tenor', 'cp_flag', 'mid_price'], keep='first', inplace=True, ignore_index=True)  # 保留距离中值最小的那个
        Options.set_index('date', inplace=True)
        # 去掉askbidspread过大的
        Options['ab_spread'] = (Options.best_offer - Options.best_bid)
        Options = Options.loc[Options.ab_spread < multiplier * Options.best_bid]

        OTM_Options = Options.loc[((Options.cp_flag == 'C') & (Options.moneynessF >= 1)) |
                                  ((Options.cp_flag == 'P') & (Options.moneynessF <= 1))]
        OTM_Options.reset_index(inplace=True)
        OTM_Options = OTM_Options.sort_values(['date', 'tenor', 'strike_price'], ascending=True)
        OTM_Options.set_index('date', inplace=True)
        # 只留下最大OTM期权

        self.OTM_Options = OTM_Options
        self.Forwards = Forwards
        self.Dividend = Dividend
        self.Interests = Interests
        self.Spots = Spots


    def getoneday(self, date, minkeep=3):
        """
        返回当天的市场基础信息, 两个期权链和其他信息, 另外需要进行数量筛选
        :param date:
        :param minkeep: 使用的期权链至少有几个OTM期权
        :return:
        """
        OTM_Option = self.OTM_Options.loc[date]
        out_put_list = []
        tenors = OTM_Option.tenor.unique()
        for tenor in tenors:
            options = OTM_Option.loc[OTM_Option.tenor == tenor]

            interest = options['interest'].unique()
            spot = options['spot'].unique()
            forward = options['forward_price'].unique()
            data = options[['otheriv', 'strike_price']].values



            out_put_list.append([np.array([tenor]),interest,spot,forward,data])

        return out_put_list


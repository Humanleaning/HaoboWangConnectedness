# 实现
# 1)输入所有需要的数据集, 然后进行数据清洗
# 2)getoneday方法, 输入日期, 获取对应数据集的日信息, 即市场信息:30天的r,q,F和两条(一条)期权链


import numpy as np


# chine是用来计算vol的, 应该从S截比较好
# chine_put是用来计算VaR的, 应该从F开始截比较好
# chine_call是用来计算VaR的, 应该从S开始截比较好

# S通过日期就能索引(index = date)
# F, q 通过日期和tenor索引(index = (date, tenor))
# r通过日期和tenor, 然后还得tenor和value线性插值(index = date), 每天肯定要计算一次r的日期, 对于大部分时间需要计算额外两次日期

def Get_r(interestset, date, tenor):
    Interest = interestset.loc[date]
    # if np.all(np.diff(Interest.tenor) > 0):
    return np.interp(tenor, Interest.tenor.values, Interest.interest.values)


# %%
# 用于提取IvyDB US的数据
class IvyDBUS_InfoExtractor(object):

    def __init__(self, Options, Forward, Interest, Spot, tenor=30):
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
        # 去掉askbidspread过大的
        Options['ab_spread'] = (Options.best_offer - Options.best_bid)
        Options = Options.loc[Options.ab_spread < 5 * Options.best_bid]
        # 去掉midprice相同的
        Options['mid_price'] = (Options.best_bid + Options.best_offer) / 2
        Options['distance_to_ATM'] = np.fabs(Options.moneynessS - 1)
        Options.reset_index(inplace=True)
        Options = Options.sort_values(['date', 'cp_flag', 'tenor', 'distance_to_ATM'], ascending=True)
        Options.drop_duplicates(subset=['date', 'tenor', 'cp_flag', 'mid_price'], keep='first',
                                inplace=True, ignore_index=True)  # 保留距离中值最小的那个
        Options.set_index('date', inplace=True)
        # 取消best bid过小的
        Options = Options.loc[Options.best_bid >= 0.01]
        # 5)数据筛选, 去掉vega过小的期权
        # Options = Options.loc[Options.vega>=0.5]

        OTM_put = Options.loc[(Options.cp_flag == 'P') & (Options.moneynessF <= 1)]
        OTM_call = Options.loc[(Options.cp_flag == 'P') & (Options.moneynessS >= 1)]
        OTM_Options = Options.loc[((Options.cp_flag == 'C') & (Options.moneynessS >= 1)) |
                                  ((Options.cp_flag == 'P') & (Options.moneynessS <= 1))]

        self.OTM_put = OTM_put
        self.OTM_call = OTM_call
        self.OTM_Options = OTM_Options

        self.Forward = Forward
        self.Interest = Interest
        self.Spot = Spot
        self.tenor = tenor

    def getoneday(self, date, minkeep=3):
        """
        ------------------------返回信息-----------------------
        返回当天的市场基础信息, 两个期权链和其他信息, 另外需要进行数量筛选


        :param date:
        :param minkeep: 使用的期权链至少有几个OTM期权
        :return:
        """
        # 获取基础信息
        OTM_Option = self.OTM_Options.loc[date]
        OTM_put = self.OTM_put.loc[date]

        S = self.Spot.loc[date, 'spot']
        F = self.Forward.loc[(date, self.tenor), 'forward_price']
        r = Get_r(self.Interest, date, self.tenor)

        # 保留至少有mink个期权的日期
        numbers = OTM_put.groupby('tenor').strike_price.count()  # 生成的是一个series
        numbers = numbers[numbers >= minkeep]
        tenors = np.array(numbers.index)
        tenors = np.sort(tenors)

        # numbers = OTM_Option.groupby(['tenor', 'cp_flag']).strike_price.count()
        # numbers = numbers>=minkeep
        # numbers = numbers.groupby(level=0).sum()
        # numbers = numbers[numbers==2]#这里的2代表两种期权同时成立
        # tenors = np.array(numbers.index)
        # tenors = np.sort(tenors)

        if (np.where(tenors == self.tenor)[0].size > 0):  # there is tenor equals 30
            chine = OTM_Option.loc[(OTM_Option.tenor == self.tenor)]
            chine_put = OTM_put.loc[(OTM_put.tenor == self.tenor)]
            return {'S': S,
                    'F': F,
                    'r': r,
                    'tenor': self.tenor,
                    'chine': chine,
                    'chine_put': chine_put,
                    'insert': False}
        else:  # there is NO tenor equals 30
            if np.where(tenors < self.tenor)[0].size > 0:  # there is a left front tenor
                tenor_front = tenors[np.where(tenors < self.tenor)[0][-1]]
                tenor_rear = tenors[np.where(tenors > self.tenor)[0][0]]
            else:  # the front tenor on right
                index = np.where(tenors > self.tenor)[0]
                tenor_front = tenors[index[0]]
                tenor_rear = tenors[index[1]]
            chine_front = OTM_Option.loc[(OTM_Option.tenor == tenor_front)]
            chine_front_put = OTM_put.loc[(OTM_put.tenor == tenor_front)]
            F_front = self.Forward.loc[(date, tenor_front), 'forward_price']
            chine_rear = OTM_Option.loc[(OTM_Option.tenor == tenor_rear)]
            chine_rear_put = OTM_put.loc[(OTM_put.tenor == tenor_rear)]
            F_rear = self.Forward.loc[(date, tenor_rear), 'forward_price']
            return {'S': S,
                    'F': F,
                    'r': r,
                    'F_front': F_front,
                    'tenor_front': tenor_front,
                    'chine_front': chine_front,
                    'chine_front_put': chine_front_put,
                    'F_rear': F_rear,
                    'tenor_rear': tenor_rear,
                    'chine_rear': chine_rear,
                    'chine_rear_put': chine_rear_put,
                    'insert': True}


# 用于提取IvyDB European的数据
# 注意, 如果是使用mid_price的版本,需要自己计算隐含波动率, 目前使用mid_price
import py_vollib_vectorized


def IVcomputer(option_chine, F, r, tenor=30):
    """
    为option_chine增加一列impl_volatility,然后返回
    :param option_chine:pandas.DataFrame 需要包含(cp_flag, strike_price, prices), 按照这个顺序排布, 命名可以随意
    :returns self.new_chine, 包含(cp_flag, strike_price, prices, impl_volatility)
    """
    annualized_tenor = tenor / 365
    option_chine['impl_volatility'] = py_vollib_vectorized.vectorized_implied_volatility_black(option_chine.mid_price,
                                                                                               F,
                                                                                               option_chine.strike_price,
                                                                                               r,
                                                                                               annualized_tenor,
                                                                                               option_chine.cp_flag.str.lower(),
                                                                                               return_as='numpy')
    return option_chine.dropna(axis=0, how='any')


class IvyDBEurope_InfoExtractor(object):

    def __init__(self, market, Options, Forwards, Dividend, Interests, Spots, tenor=30):
        """

        :param market: str 要计算oi的市场
        :param Options:
        :param Forwards:
        :param Interests:
        :param Spots:
        :param tenor:  要计算oi information的tenor
        """
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
        Options.drop_duplicates(subset=['date', 'tenor', 'cp_flag', 'mid_price'], keep='first',
                                inplace=True, ignore_index=True)  # 保留距离中值最小的那个
        Options.set_index('date', inplace=True)
        # 去掉askbidspread过大的
        Options['ab_spread'] = (Options.best_offer - Options.best_bid)
        Options = Options.loc[Options.ab_spread < multiplier * Options.best_bid]

        # 只留下最大OTM期权

        self.Options = Options
        self.Forwards = Forwards
        self.Dividend = Dividend
        self.Interests = Interests
        self.Spots = Spots
        self.tenor = tenor

    def getoneday(self, date, minkeep=3):
        """
        返回当天的市场基础信息, 两个期权链和其他信息, 另外需要进行数量筛选
        :param date:
        :param minkeep: 使用的期权链至少有几个OTM期权
        :return:
        """
        # 获取基础信息
        Option = self.Options.loc[date]
        S = self.Spots.loc[date, 'spot']
        F = self.Forwards.loc[(date, self.tenor), 'forward_price']
        r = Get_r(self.Interests, date, self.tenor)

        # # 保留至少有mink个期权的日期
        # numbers = Option.loc[(Option.cp_flag == 'P') & (Option.moneynessS < 1.02)].groupby(
        #     'tenor').strike_price.count()  # 生成的是一个series
        # numbers = numbers[numbers >= minkeep]
        # tenors = np.array(numbers.index)
        # tenors = np.sort(tenors)

        numbers = Option.loc[((Option.cp_flag == 'P') & (Option.moneynessS <= 1))|
                             ((Option.cp_flag == 'C') & (Option.moneynessS >  1))].groupby(['tenor', 'cp_flag']).strike_price.count()
        numbers = numbers >= minkeep
        numbers = numbers.groupby(level=0).sum()
        numbers = numbers[numbers == 2]#这里的2代表两种期权同时成立
        tenors = np.array(numbers.index)
        tenors = np.sort(tenors)


        # 按照是否正好有self.tenor的期权链来返回数据
        if np.where(tenors == self.tenor)[0].size > 0:  # there is tenor equals 30

            chine = Option.loc[(Option.tenor == self.tenor)]
            chine['moneynessF'] = chine.strike_price / F
            chine = chine.loc[((chine.cp_flag == 'C') & (chine.moneynessF >= 1)) |
                              ((chine.cp_flag == 'P') & (chine.moneynessF <= 1))]
            chine = IVcomputer(chine, F, r, tenor=self.tenor)
            chine_put = chine.loc[chine.cp_flag == 'P']
            # chine_call = chine.loc[chine.cp_flag == 'C']

            return {'S': S,
                    'F': F,
                    'r': r,
                    'tenor': self.tenor,
                    'chine': chine,
                    'chine_put': chine_put,
                    # 'chine_call': chine_call,
                    'insert': False}

        else:  # there is NO tenor equals 30
            if np.where(tenors < self.tenor)[0].size > 0:  # there is a left front tenor
                tenor_front = tenors[np.where(tenors < self.tenor)[0][-1]]
                tenor_rear = tenors[np.where(tenors > self.tenor)[0][0]]
            else:  # the front tenor on right
                index = np.where(tenors > self.tenor)[0]
                tenor_front = tenors[index[0]]
                tenor_rear = tenors[index[1]]

            chine_front = Option.loc[(Option.tenor == tenor_front)]
            q1 = self.Dividend.loc[(date, tenor_front), 'dividend']
            r1 = Get_r(self.Interests, date, tenor_front)
            F_front = S * np.exp((r1 - q1) * tenor_front / 365)
            chine_front['moneynessF'] = chine_front.strike_price / F_front
            chine_front = chine_front.loc[((chine_front.cp_flag == 'C') & (chine_front.moneynessF >= 1)) |
                                          ((chine_front.cp_flag == 'P') & (chine_front.moneynessF <= 1))]
            chine_front = IVcomputer(chine_front, F_front, r1, tenor=tenor_front)
            chine_front_put = chine_front.loc[chine_front.cp_flag == 'P']
            # chine_front_call = chine_front.loc[chine_front.cp_flag == 'C']

            chine_rear = Option.loc[(Option.tenor == tenor_rear)]
            q2 = self.Dividend.loc[(date, tenor_rear), 'dividend']
            r2 = Get_r(self.Interests, date, tenor_rear)
            F_rear = S * np.exp((r2 - q2) * tenor_rear / 365)
            chine_rear['moneynessF'] = chine_rear.strike_price / F_rear
            chine_rear = chine_rear.loc[((chine_rear.cp_flag == 'C') & (chine_rear.moneynessF >= 1)) |
                                        ((chine_rear.cp_flag == 'P') & (chine_rear.moneynessF <= 1))]
            chine_rear = IVcomputer(chine_rear, F_rear, r2, tenor=tenor_rear)
            chine_rear_put = chine_rear.loc[chine_rear.cp_flag == 'P']
            # chine_rear_call = chine_rear.loc[chine_rear.cp_flag == 'C']

            return {'S': S,
                    'F': F,
                    'r': r,
                    'F_front': F_front,
                    'tenor_front': tenor_front,
                    'chine_front': chine_front,
                    'chine_front_put': chine_front_put,
                    # 'chine_front_call': chine_front_call,
                    'F_rear': F_rear,
                    'tenor_rear': tenor_rear,
                    'chine_rear': chine_rear,
                    'chine_rear_put': chine_rear_put,
                    # 'chine_rear_call': chine_rear_call,
                    'insert': True}

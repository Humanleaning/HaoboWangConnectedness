# -*- coding: utf-8 -*-


import pandas as pd
import scipy.stats as stats
import arch.unitroot

class description():
    """
    创建统计性描述类
    """
    def __init__(self, opt, decimal=4):
        """
        初始化
        :param opt: 统计量顺序列表
        :param decimal: 保留小数点位数
        """
        self.stat = ['Mean', 'Median', 'Mode', 'Maximum', 'Minimum', 'Vanriance', 'Std.',
                     'Skew.', 'Kurt.', 'J-B', 'ADF', 'PP', 'KPSS', 'Obs.', 'Sum']
        self.opt = opt
        self.decimal = decimal

    def run(self, df, save_path='output.xlsx'):
        """
        串接整个流程:
            1. 计算全部统计量
            2. 按顺序选择对应统计量
            3. 保存为excel数据
        :param df: 待处理数据，为pandas.DataFrame格式
        :param save_path: 保存路径
        """
        output = self.calculate_statistic(df)
        out = self.sorted_statistic(output)
        decimal_str = f'%.{self.decimal}f'
        out.to_excel(save_path, index=list(
            df.columns), float_format=decimal_str)

    def significance_level(self, statistic, p, decimal=3):
        """
        根据p值确定显著水平，并进行标星，其中1%水平为***，5%水平为**，10%水平为*
        :param statistic: 统计量值
        :param p: 统计量p值
        :param decimal: 保留小数点位数
        :return: 统计量标星
        """
        statistic = round(statistic, decimal)
        if p <= 0.01:
            return f'{statistic}***'
        elif p <= 0.05:
            return f'{statistic}**'
        elif p <= 0.1:
            return f'{statistic}*'
        else:
            return str(statistic)

    def calculate_statistic(self, df):
        """
        计算全部统计量
        :param df: 待处理数据
        :return output: 全部统计量
        """
        output = pd.DataFrame(index=self.stat)
        for column in df.columns:
            data = df[column]
            stat_value = [data.mean(), data.median(), data.mode()[0], data.max(
            ), data.min(), data.var(), data.std(), data.skew(), data.kurt()]
            statistic, p = stats.jarque_bera(data)   # JB检验
            stat_value.append(self.significance_level(
                statistic, p, self.decimal))
            a = arch.unitroot.ADF(data)
            stat_value.append(self.significance_level(
                a.stat, a.pvalue, self.decimal))   # ADF检验
            a = arch.unitroot.PhillipsPerron(data)
            stat_value.append(self.significance_level(
                a.stat, a.pvalue, self.decimal))   # PP检验
            a = arch.unitroot.KPSS(data)
            stat_value.append(self.significance_level(
                a.stat, a.pvalue, self.decimal))   # KPSS检验
            stat_value += [len(data), data.sum()]
            output[column] = stat_value
        return output.T

    def sorted_statistic(self, output):
        """
        按顺序选择对应统计量
        :param output: 待处理的全部统计量
        :return: 有顺序的统计量
        """
        opt_stat = []
        [opt_stat.insert(j-1, i)
         for i, j in zip(self.stat, self.opt) if j != 0]
        return output[opt_stat]


def deal_int(data):
    """
    处理GUI界面中各顺序量
    :param data: 顺序数字，列表形式
    :return out: 有则返回对应整数，无则返回0,形成的列表
    """
    out = []
    for i in data:
        try:
            out.append(int(i))
        except:
            out.append(0)
    return out

#%%
if __name__ == '__main__':
    dataurl = r'DataDescription\RIX.csv'
    first_column_is_date = True
    opt = list(range(1, 16))
    stat = description(opt, decimal=6)
    df = pd.read_csv(dataurl)
    if first_column_is_date:
        df = df.drop(df.columns[0], axis=1)   # 删除第一列日期数据
    stat.run(df)

#在统计学中，Jarque–Bera检验是对样本数据是否具有符合正态分布的偏度和峰度的拟合优度的检验。其统计测试结果总是非负的。如果结果远大于零，则表示数据不具有正态分布。
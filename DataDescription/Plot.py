import pandas as pd
import numpy as np



#%%
# 这里请讲地址替换成你自己的文件, 注意需要是csv格式, 字符串r表示raw string, 防止转义符发挥作用,不要去掉
# 数据第一列需要时日期

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
sns.set_context("paper")

fdata = pd.read_csv(r'../results/SectorVaR05.csv', parse_dates=[0], index_col=0)
markets = list(fdata.columns)
print(markets)


#%%
#这里画第一张图片
group0 = ['GLD', 'USO']
group1 = ['XLB', 'XLE']
group2 = ['XLF', 'XLI', 'XLK']
group3 = ['XLP', 'XLU']
group4 = ['XLV', 'XLY']
group0_name = '1) 中南喜剧人'
group1_name = '2) 中南艺术家'
group2_name = '3) 中南任你挑'
group3_name = '4) 中南哈哈哈'
group4_name = '5) 中南带回家'
#%%
####图1-4
fig,axes=plt.subplots(2,2,figsize=(15,10),sharex=False,sharey=False)#sharex,sharey表示是否公用坐标轴
for n in range(0,4):
    i,j = divmod(n,2)
    group = eval('group'+str(n))
    group_name = eval('group'+str(n)+'_name')
    data = fdata[group]
    for market in group:
        axes[i,j].plot(data.index, data[market],linewidth=0.2, label=market)

    axes[i,j].legend(bbox_to_anchor=(0.5, -0.15), loc='lower center', ncol=len(group), borderaxespad=0.)
    axes[i,j].set_title(group_name, loc='left')
plt.tight_layout()
plt.show()
#%%
####图5
plt.figure(figsize=(16,9))
group = group4
group_name = group4_name
data = fdata[group]
for market in group:
    plt.plot(data.index, data[market], linewidth=0.2, label=market)
plt.legend(bbox_to_anchor=(0.5, -0.1
                           ), loc='lower center', ncol=len(group), borderaxespad=0.)
plt.title(group_name, loc='left')
plt.show()




#%%
#这里画第三个图片, 如果想要更改具体的图片参数, 请查看函数
def correlation_matrix(data, fullsample = True, start=None, end=None):
    if fullsample:
        pass
    else:
        data = data.loc[start:end]
    plt.rcParams['figure.figsize'] = (17, 15)
    sns.heatmap(data.corr(),
                annot=True,
                linewidths=.5,
                cmap="RdBu_r",  ##############这里调整颜色, 故意输错一下这个参数可以显示所有可行颜色
                annot_kws={"size": 25})
    sns.set(font_scale=2)
    plt.xticks(fontsize=30,  # 字体大小
               rotation=45,  # 字体是否进行旋转
               horizontalalignment='right'  # 刻度的相对位置
               )
    plt.yticks(fontsize=30)
    plt.tight_layout()
    # plt.title('Correlation between features', fontsize = 30)
    plt.show()

#%%
#correlation_matrix(fdata)#全样本
correlation_matrix(fdata, fullsample=False, start='2018-1-1', end='2020-1-1')









import numpy as np
import pandas as pd
def Rf(Time):
    data = pd.read_csv('2021年中债国债收益率曲线标准期限信息.csv', encoding='gbk')
    RF = []
    for i in range(len(data)):
        if data['标准期限(年)'][i] == Time :
            RF.append(data['收益率(%)'][i]/100)
    RF = np.array(RF).mean()
    return RF

def price(Price, Strike, Rate, Time, Increment, Volatility, Flag):
    """
	输入参数及其解释：
	price:标的资产市场价格
	strike：执行价格
	rate：无风险利率
	time：距离到期时间
	increment：每个阶段的时间间隔，如一年可分为12阶二叉树
	volatility：波动率
	flag：标记期权种类，1为看涨期权，2为看跌期权
	"""
    Price, Strike, Rate, Time, Volatility = float(Price), float(Strike), float(Rate), float(Time), float(Volatility)
    u = np.exp(Volatility * np.sqrt(Increment))
    d = 1 / u
    p = (np.exp(Rate*Increment) - d) / (u - d)
    N = int(Time/Increment)
    # 初始化
    AssetPrice = np.zeros([N+1, N+1])
    OptionValue = np.zeros([N+1, N+1])
    # 计算资产二叉树的价值
    for i in range(0, N+1):
        AssetPrice[i][i] = Price * d ** i
        for j in range(i, N):
            AssetPrice[i][j+1] = AssetPrice[i][j] * u
    # 计算期权二叉树的价值
    for i in range(0, N+1):
        if Flag == 1:
            OptionValue[i][N] = max(AssetPrice[i][N] - Strike, 0)
        elif Flag == 2:
            OptionValue[i][N] = max(Strike - AssetPrice[i][N], 0)

    for i in range(0, N):
        for j in range(i, N):
            OptionValue[N-1-j][N-1-i] = np.exp(-Rate*Increment) * (p*OptionValue[N-1-j][N-i] + (1-p)*OptionValue[N-j][N-i])
    '''
    AssetPrice:基础资产价格变化过程
    OptionValue:期权价格变化过程
    '''
    return AssetPrice, OptionValue


if __name__ == "__main__":

   Price = 100 
   Strike = 95
   Time = 0.25
   Rate = Rf(Time)
   Increment = 1/12
   Volatility = 0.5 
   # 看涨期权
   AssetPrice , OptionValue = price(Price, Strike, Rate, Time, Increment, Volatility, 1)
   print('基础资产价格变化')
   print(AssetPrice,'\n')
   print('看涨期权价格变化')
   print(OptionValue,'\n')
   # 看跌期权
   AssetPrice , OptionValue = price(Price, Strike, Rate, Time, Increment, Volatility, 2)
   print('基础资产价格变化')
   print(AssetPrice,'\n')
   print('看跌期权价格变化')
   print(OptionValue,'\n')
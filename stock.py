import pandas as pd
import akshare as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style  as style
from mplfinance.original_flavor import candlestick2_ohlc
from matplotlib.ticker import FormatStrFormatter
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from datetime import datetime
import matplotlib.dates as mdates
import scipy.optimize as sco
import math



'''
金融数量分析计算框架
stock 主要实现股票技术分析数据制作、作图、期望收益率计算
frontcon 实现组合前缘的计算
plan 具体实现投资计划
'''
'''
以下为设置区
'''
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] #设置显示图片中文字符

'''
 计算无风险利率
 这里采用0年期(短期)国债平均收益率
 数据来源(截止至2021/4/21)：
 http://yield.chinabond.com.cn/cbweb-mn/yc/downYearBzqxList?wrjxCBFlag=0&&zblx=txy&&ycDefId=2c9081e50a2f9606010a3068cae70001

''' 
data=pd.read_csv('2021年中债国债收益率曲线标准期限信息.csv',encoding='gbk')
RF=[]
for i in range(len(data)):
    if data['标准期限(年)'][i] == 0 :
       RF.append(data['收益率(%)'][i]/100)
RF=np.array(RF).mean()


'''
以下为框架主体
'''

class stock():
	__n_periods__ = 10 #设置预测周期
	def __init__(self,dataframe,name,day):
		self.df = dataframe
		self.name = name
		self.day = day

	def MA(self):#设置k线图显示天数
		'''
		计算技术分析数据
		'''
		data = self.df
		day=self.day
		df3 = data.reset_index().iloc[-day:,:6]  #取过去n天数据
		df3 = df3.dropna(how='any').reset_index(drop=True) #去除空值且从零开始编号索引
		df3 = df3.sort_values(by='date', ascending=True) #df3是原始数据
		stdev_factor=2
		df3['5'] = df3.close.rolling(5).mean()#五日均线
		df3['10'] = df3.close.rolling(10).mean()#十日均线
		df3['20'] = df3.close.rolling(20).mean()#二十日均线
		df3['upper_band']=df3.close.rolling(20).mean()+stdev_factor*df3.close.rolling(20).std()
		df3['lower_band']=df3.close.rolling(20).mean()-stdev_factor*df3.close.rolling(20).std()
		df3.dropna(axis=0,inplace=True)
		return(df3)

	def k_plot(self):
		'''
		画K线图
		5、10、20日均线
		布林带
		'''
		df3=self.MA()
		name=self.name
		style.use('ggplot')
		fig, ax = plt.subplots(1, 1, figsize=(8,3), dpi=200)
		candlestick2_ohlc(ax,
                opens = df3[ 'open'].values,
                highs = df3['high'].values,
                lows = df3[ 'low'].values,
                closes = df3['close'].values,
                width=0.5, colorup="r",colordown="g")
		ax.text( df3.high.idxmax()-20, df3.high.max(),s =df3.high.max(), fontsize=8)
		ax.text( df3.high.idxmin()-20, df3.high.min(),s = df3.high.min(), fontsize=8)
		ax.set_facecolor("white")
		ax.set_title(name)
		plt.plot(df3['5'].values, alpha = 0.5, label='MA5')
		plt.plot(df3['10'].values, alpha = 0.5, label='MA10')
		plt.plot(df3['20'].values, alpha = 0.5, label='MID(MA20)',color='gray')
		plt.plot(df3['upper_band'].values, alpha = 0.5, label='UPR',color='yellow')
		plt.plot(df3['lower_band'].values, alpha = 0.5, label='DN',color='blue')
		ax.legend(facecolor='white', edgecolor='white', fontsize=6)
		plt.xticks(ticks =  np.arange(0,len(df3)), labels = df3.date.dt.strftime('%Y-%m-%d').to_numpy() )
		plt.xticks(rotation=90, size=8)
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
		plt.show()

	def forcast_model(self):
		'''
		使用arima模型进行收盘价的预测
		'''
		df1=self.MA()
		df1=pd.DataFrame(df1['close'])
		df1.index=[i for i in range(len(df1))]
		df=df1.close
		model = pm.auto_arima(df, start_p=1, start_q=1,
                      information_criterion='aic',
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=5, max_q=5, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
		return(model)

	def forcast_plot(self):
		df1=self.MA()
		df1=pd.DataFrame(df1['close'])
		df1.index=[i for i in range(len(df1))]
		df=df1.close
		plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
		fig, axes = plt.subplots(3, 3, sharex=False)
		plt.subplots_adjust(wspace=0.2, hspace=0.3)
		plt.rc('font', size=5)
		# Original Series
		axes[0, 0].plot(df); axes[0, 0].set_title('原序列')
		plot_acf(df, ax=axes[0, 1])
		plot_pacf(df, ax=axes[0, 2])
		# 1st Differencing
		axes[1, 0].plot(df.diff()); axes[1, 0].set_title('一阶差分序列')
		plot_acf(df.diff().dropna(), ax=axes[1, 1])
		plot_pacf(df.diff().dropna(), ax=axes[1, 2])
		# 2nd Differencing
		axes[2, 0].plot(df.diff().diff()); axes[2, 0].set_title('二阶差分序列')
		plot_acf(df.diff().diff().dropna(), ax=axes[2, 1])
		plot_pacf(df.diff().diff().dropna(), ax=axes[2, 2])
		#Display pic
		plt.show()
		df.reindex([i for i in range(len(df))])
		model = self.forcast_model()
		print(model.summary())
		# Forecast
		fc, confint = model.predict(n_periods=self.__n_periods__, return_conf_int=True)
		index_of_fc = np.arange(len(df), len(df)+self.__n_periods__)
		# make series for plotting purpose
		fc_series = pd.Series(fc, index=index_of_fc-1)
		lower_series = pd.Series(confint[:, 0], index=index_of_fc-1)
		upper_series = pd.Series(confint[:, 1], index=index_of_fc-1)
		# plot resid
		residuals = pd.DataFrame(model.resid())
		fig, ax = plt.subplots(1,2)
		residuals.plot(title="残差图", ax=ax[0])
		residuals.plot(kind='kde', title='正态检验图', ax=ax[1])
		plt.show()
		plt.plot(df,label='实际值')
		plt.plot(fc_series, color='darkgreen',label='预测均值')
		plt.legend(loc ='best')
		plt.fill_between(lower_series.index, 
		                 lower_series, 
		                 upper_series, 
		                 color='k', alpha=.15)
		#作图
		plt.rc('font', size=7)
		plt.title("%s收盘价预测"%self.name)
		plt.xlim(1,len(df)+5)
		plt.ylim(df[len(df)-1]-5,df[len(df)-1]+5)
		return(fc_series)

	def ExpReturn1(self):
		'''
		使用时间序列方法计算期望收益率
		'''
		model = self.forcast_model()
		print(model.summary())
		# Forecast
		fc, confint = model.predict(n_periods=self.__n_periods__, return_conf_int=True)
		fc_series = pd.Series(fc)		
		forcast=fc_series.iloc[0]
		now=self.df['close'][len(self.df)-1]
		rate=(forcast-now)/now
		#print('期望收益率为%s'%rate)
		return(rate)

	def ExpReturn2(self):
		'''
		历史收益率均值,返回一个值
		'''
		df1=self.df
		df1=pd.DataFrame(df1['close'])
		df1.index=[i for i in range(len(df1))]
		df=df1.close
		Return=[]
		for i in range(len(df)-1):
			Return.append((df[i+1]-df[i])/df[i])
		Return=np.array(Return).mean()
		#print('期望收益率为%s'%Return)
		return(Return)	

	def His_Return(self):
		'''
		所有历史收益率,返回一个列表
		'''
		n=900 #取过去n个交易日数据
		df1=self.df.reset_index().iloc[-n:,:6]  
		df1=pd.DataFrame(df1['close'])
		df1.index=[i for i in range(len(df1))]
		df=df1.close
		Return=[]
		for i in range(len(df)-1):
			Return.append((df[i+1]-df[i])/df[i])
		return(np.array(Return))
	
	@ property
	def sharp_rate(self):
		ER = self.His_Return().mean() # 资产的平均收益率
		sigma = self.His_Return().std() # 资产收益率的标准差
		rate = (ER-RF)/sigma
		return rate



'''
求解马科维茨前缘组合
'''
def frontcon(ExpReturn, ExpCovariance, NumPorts=10):
    noa = len(ExpReturn)

    def statistics(weights):
        weights = np.array(weights)
        z = np.dot(ExpCovariance, weights)
        x = np.dot(weights, z)
        port_returns = (np.sum(ExpReturn * weights.T))
        port_variance = np.sqrt(x)
        num1 = port_returns / port_variance
        return np.array([port_returns, port_variance, num1])

    # 定义一个函数对 方差进行最小化
    def min_variance(weights):
        return statistics(weights)[1]

    bnds = tuple((0, 1) for x in range(noa))
    # 在不同目标收益率水平（target_returns）循环时，最小化的一个约束条件会变化。
    target_returns = np.linspace(min(ExpReturn), max(ExpReturn), NumPorts)
    target_variance = []
    PortWts = []
    for tar in target_returns:
        # 在最优化时采用两个约束，1.给定目标收益率，2.投资组合权重和为1。
        cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - tar}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        res = sco.minimize(min_variance, noa * [1. / noa, ], method='SLSQP', bounds=bnds, constraints=cons)
        target_variance.append(res['fun'])
        PortWts.append(res["x"])
    target_variance = np.array(target_variance)
	# 有效前沿
    postive_target_returns=[]
    postive_target_variance=[]
    for i in range(len(PortWts)-1):
        if target_variance[i+1]>target_variance[i] :
            postive_target_returns.append(target_returns[i])
            postive_target_variance.append(target_variance[i])
    postive_target_returns.append(target_returns[-1])
    postive_target_variance.append(target_variance[-1])
    return [target_variance, target_returns, PortWts, postive_target_returns, postive_target_variance]

# 计算前沿组合的证券市场线及风险溢价  
def Get_M(postive_target_returns,postive_target_variance,RF):
    # 定义一个函数用来 计算 前后两点在 证券市场线上的值
    def value(k,RF,x):
        return k*x+RF
    # 计算 M 点
    K=[]
    for i in range(len(postive_target_variance)):
        k = (postive_target_returns[i]-RF)/postive_target_variance[i]
        K.append(k)
    
    for i in range(len(postive_target_variance)): 
        k=K[i]   
        if i > 0 and i < len(postive_target_variance)-1 :
            xl=postive_target_variance[i-1]
            yl=postive_target_returns[i-1]
            xr=postive_target_variance[i+1]
            yr=postive_target_returns[i+1]
            if value(k,RF,xl) > yl  and value(k,RF,xr) > yr :
                M = i
        elif i == 0 or i == len(postive_target_variance) :
            M=K.index(max(K)) 
    E_rm = K[M]
    return [M,E_rm]

# 计算资本市场线上的任意投资组合P的预期风险	
def sigma_rp(ExpCov,PortWts,m):
	'''
	sigma(r_p)=[ ∑ ∑ w_i w_j cov( r_i , r_j ) ]^(1/2)
	'''
	sigma_r_p=0
	w=PortWts[m]
	for i in range(len(ExpCov)):
		for j in range(len(ExpCov)):
			cov_ri_rj = ExpCov[i][j]
			sigma_r_p += w[i]*w[j]*cov_ri_rj
	return sigma_r_p

# 计算组合的beta系数
def beta(sigma_rp,singma_rm): 
	'''
	sigma_rp 是资本市场线是 资本市场组合 的预期风险
	sigma_rm 是组合前沿上 m点的 风险组合 的预期风险

	（2）该证券对风险资产市场组合风险的贡献
		程度，也就是系统性风险系数 ，这也是决定
		该项证券期望收益的关键因素。
		beta>1：该证券的风险补偿大于市场组合的风
		险补偿 (进取型证券)
		beta<1：该证券的风险补偿小于市场组合的风
		险补偿 （防御型证券)

	由 beta_p = ∑ w_i*beta_i 可以得出每支股票的beta值 
	'''
	return sigma_rp/singma_rm


'''
按照选择的组合计算股票购买量
'''
class plan():
	def __init__(self,stock_list,PortWts,assets_amount):
		self.stock_list = stock_list
		self.PortWts = PortWts
		self.assets_amount = assets_amount

	def perchase_plan(self):
		for i in range(len(self.stock_list)):
			stock=self.stock_list[i]
			df = ak.stock_zh_a_daily(symbol=stock, adjust="qfq")
			money=self.PortWts[i]*self.assets_amount
			amount=math.floor(money/(df['close'][len(df)-1]*100))*100
			print('股票%s购买%s股'%(stock,amount))
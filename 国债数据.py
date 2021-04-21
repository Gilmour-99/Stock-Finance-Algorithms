import requests
import pandas as pd

def get_request(url):
    headers={"user-agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv,2.0.1) Gecko/20100101 Firefox/4.0.1"}
    response = requests.get(url=url,headers=headers)
    if response.status_code == 200 :
        dic = response
        print('下载成功')
    else : 
        print('Error')
    return(dic)

dic=get_request('http://yield.chinabond.com.cn/cbweb-mn/yc/downYearBzqx?year=2021&&wrjxCBFlag=0&&zblx=txy&&ycDefId=2c9081e50a2f9606010a3068cae70001&&locale=')

data=pd.ExcelFile(dic.content)
table1=data.parse(data.sheet_names[0])
table1.to_csv('2021年中债国债收益率曲线标准期限信息.csv',encoding='gbk')
print('表格保存成功')
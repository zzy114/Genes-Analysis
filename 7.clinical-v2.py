# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 11:31:11 2018

@author: zzy
"""
#30.预后分析数据导入
from pandas import Series,DataFrame
import pandas as pd
import numpy as np

#31.clinical数据预处理
clinical_LUAD = pd.read_csv('C:\\Users\\admin\\Desktop\\zzy\\lung ML\\clinical_info_LUAD.csv',index_col=0,header=0,engine='python',encoding=None,chunksize=None)
clinical_LUSC = pd.read_csv('C:\\Users\\admin\\Desktop\\zzy\\lung ML\\clinical_info_LUSC.csv',index_col=0,header=0,engine='python',encoding=None,chunksize=None)
def clinical(data,type):
    data = data.iloc[:,[5,7,8,9]]
    data = data.where(data.notnull(), 0)
    for i in range(data.shape[0]):
        if int(data.iloc[i,[2]]) == 0:
            if int(data.iloc[i,[1]]) == 0:
                data.iloc[i,[2]] = int(data.iloc[i,[3]])
            else:
                data.iloc[i,[2]] = int(data.iloc[i,[1]])
    data = data.replace('Alive',0)
    data = data.replace('Dead',1)
    data = data.iloc[:,[2,0]]
    data.iloc[:,0] = data.iloc[:,0]/365
    data['type']=type
    return data

clinical_LUAD = clinical(clinical_LUAD,'LUAD')
clinical_LUSC = clinical(clinical_LUSC,'LUSC')
#拼接
clinical = pd.concat([clinical_LUAD, clinical_LUSC])

#32.预后数据和癌症表达值数据的样本量,id及顺序不同，需要进行查找和排序
#预测值读入
pred = pd.read_table('C:\\Users\\admin\\Desktop\\zzy\\lung ML\\pred.txt',header=None,encoding='gb2312',delim_whitespace=True,index_col=0)
#预测结果id与clinical中id进行格式匹配
index=[]

for i in pred.index:
    index.append(i.replace('.','-')[:12])
    
pred.index = index

#样本匹配
clinical1=DataFrame()

for i in pred.index:
    count = -1
    for j in clinical.index:
        count+=1
        if i == j:
            clinical1 = pd.concat([clinical1,clinical.iloc[count,:]],axis=1)
            break
        
clinical1=clinical1.T        
        
pred = pred.replace(1,'LUAD')
pred = pred.replace(0,'LUSC')
pred.columns=['type']
pred = pd.concat([clinical1.iloc[:,[0,2]],pred],axis=1)
 

#33.生存曲线作图, pip install lifelines
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

def lifelines_plt(data,type1,type2,title,path):
    kmf = KaplanMeierFitter()
    T = data['days_to_death']
    E = data['vital_status']
    groups = data['type']
    ix = (groups == type1)
     
    kmf.fit(T[~ix], E[~ix], label=type2)
    ax = kmf.survival_function_.plot()
     
    kmf.fit(T[ix], E[ix], label=type1)
    kmf.survival_function_.plot(ax=ax)
    
    plt.ylabel("Percentage")
    plt.xlabel('Year')
    plt.title(title)
    plt.savefig(path)

lifelines_plt(pred,'LUAD','LUSC',u'Lifelines of Prediction',r'C:\\Users\\admin\Desktop\\zzy\\lung ML\\pred.pdf')  
lifelines_plt(clinical1,'LUAD','LUSC',u'Lifelines of Clinical',r'C:\\Users\\admin\Desktop\\zzy\\lung ML\\clinical.pdf')  

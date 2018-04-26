# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 11:45:46 2018

@author: admin
"""

from pandas import Series,DataFrame


import pandas as pd
import numpy as np
#1.数据导入
data1 = pd.read_table('C:\\Users\\admin\\Desktop\\zzy\\lung ML\\TCGA_LUAD_mRNA_counts.txt',header=None,encoding='gb2312',delim_whitespace=True,index_col=0)
data2 = pd.read_table('C:\\Users\\admin\\Desktop\\zzy\\lung ML\\TCGA_LUSC_mRNA_counts.txt',header=None,encoding='gb2312',delim_whitespace=True,index_col=0)

#2.正则匹配
import re
def match01A(data):
    b = data.iloc[0,:]
    count = 0
    for i in b:
        count += 1
        temp = str(i)
        r = re.search(r'01A',temp)
        if r == None:
            data.drop(count, inplace = True,axis=1)
    #索引第一列命名为id
    a = np.array(data1.index)
    a[0]='id'
    data.index=a
    return data

data1=match01A(data1)
data2=match01A(data2)

#导出
data1.to_csv('C:\\Users\\admin\\Desktop\\zzy\\lung ML\\LUAD_mRNA.txt',header = False,sep = '\t')
data2.to_csv('C:\\Users\\admin\\Desktop\\zzy\\lung ML\\LUSC_mRNA.txt',header = False,sep = '\t')
#之后使用R语言标准化和差异分析




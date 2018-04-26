# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 18:40:10 2018

@author: zhiyang
"""
from sklearn.metrics import accuracy_score,roc_auc_score 
from pandas import Series,DataFrame
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn import neighbors
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#接在2.normalization&4.mRNA analysis之后
#12.数据导入
path=r'C:\Users\admin\Desktop\zzy\lung ML\\probeid.FDR0.05.exprs.txt'
df=pd.read_table(path,index_col=0,header=0,engine='python',encoding=None,chunksize=None)
X = df.T
Y=Series(np.zeros(X.shape[0]))
Y[0:519]=1

#13.测试集、训练集划分

def split_train_test(X,Y, size=0.2):
    X = DataFrame(X)
    Y = Series(Y)
    folds = int(1/size)
    kfold=StratifiedKFold(Y,n_folds=folds,random_state=1)  
    x_train = DataFrame()  
    y_test = Series()   
    y_train = Series()   
    x_test = DataFrame() 
    for train, test in kfold:   
        x_train = x_train.append(X.iloc[train]) 
        y_train = y_train.append(Y.iloc[train])
        y_test = y_test.append(Y.iloc[test])
        x_test = x_test.append(X.iloc[test])
        break  
    return x_train,y_train,x_test,y_test

x_train,y_train,x_test,y_test = split_train_test(X,Y)

#14.基于Logistic的Lasso降维
from sklearn.feature_selection import SelectFromModel
log_reg = LogisticRegression(penalty='l2',C=6,random_state= 1)
log_reg.fit(x_train, y_train)
model = SelectFromModel(log_reg, prefit=True)
x_train1 = model.transform(x_train)
x_test1 = model.transform(x_test)

#15.基于Tree importance筛选基因
gbdt=GradientBoostingClassifier(n_estimators=170,learning_rate=0.1,subsample= 0.5,random_state=1)
gbdt.fit(x_train1, y_train)
weights=gbdt.feature_importances_
weights_sort=np.argsort(weights)[::-1]
selected = weights_sort[:50]
x_train2 = x_train1[:,selected]
x_test2 = x_test1[:,selected]

#16.包裹法，sbs降维
from sklearn.base import clone
from itertools import combinations
class SBS():
    def __init__(self,estimator,k_features,scoring=roc_auc_score,test_size=0.2,random_state=1):
        self.scoring=scoring
        self.estimator=clone(estimator)
        self.k_features=k_features
        self.test_size=test_size
        self.random_state=random_state
        self.a = []
        self.b = []
    def fit(self,x,y):
        x_train,y_train,x_test,y_test = split_train_test(x,y,size=self.test_size)
        x_train,y_train,x_test,y_test = x_train.values,y_train.values,x_test.values,y_test.values
        dim=x_train.shape[1]
        self.indices_=tuple(range(dim))
        self.subsets_=[self.indices_]
        score=self._cal_score(x_train,y_train,x_test,y_test,self.indices_)
        self.scores_=[score]
        while dim >self.k_features:
            scores=[]
            subsets=[]
            for p in combinations(self.indices_,r=dim-1):
                score=self._cal_score(x_train,y_train,x_test,y_test,p)
                scores.append(score)
                subsets.append(p)
            best=np.argmax(scores)
            
            self.a.append(dim-1)
            self.b.append(scores[best])
            
            print(dim-1,'维分数为：',scores[best])
            self.indices_=subsets[best]
            self.subsets_.append(self.indices_)
            dim-=1
            self.scores_.append(scores[best])
        return self 

    def transform(self,x,subset):
        return x[:,subset]
    
    def subsets(self):
        return self.subsets_
    
    #返回维度和分数，用于作图
    def plt(self):
        return self.a,self.b

    def _cal_score(self,x_train,y_train,x_test,y_test,indices):
        self.estimator.fit(x_train[:,indices],y_train)
        y_pred=self.estimator.predict(x_test[:,indices])
        score=self.scoring(y_test,y_pred)
        return score
    

#17.基于GBDT的SBS降维
sbs = SBS(estimator = gbdt, k_features = 5)
sbs.fit(x_train2,y_train)
subset = sbs.subsets()[(50-12)]
#print (subset)  <<<(0, 1, 2, 4, 7, 9, 10, 13, 17, 18, 22, 23)
x_train3 = sbs.transform(x_train2,subset)
x_test3 = sbs.transform(x_test2,subset)

#18.模型训练
#GBDT，必然比单纯的随机森林好，随机森林不再测试
#网格搜索
param_test1 = [{'n_estimators':list(range(10,201,10)),'subsample':[x/10 for x in range(1,10)]}]
gs1 = GridSearchCV(estimator = GradientBoostingClassifier(),param_grid = param_test1, n_jobs=-1,scoring='roc_auc',iid=False,cv=5)
gs1.fit(x_train3, y_train)
gs1.grid_scores_, gs1.best_params_, gs1.best_score_  

gbdt=GradientBoostingClassifier(n_estimators=190,learning_rate=0.1,subsample= 0.4,random_state=1)
gbdt.fit(x_train3, y_train)
pred1 = gbdt.predict(x_test3)
#准确率
roc_score1 = roc_auc_score(y_test, pred1)  
acc_score1 = accuracy_score(y_test, pred1) 


#Logistic回归
#网格搜索
param_test2 = [{'penalty':['l1','l2'],'C':list(range(1,10))}]
gs2 = GridSearchCV(estimator = LogisticRegression(random_state= 1),param_grid = param_test2, scoring='roc_auc',iid=False,cv=5,n_jobs=-1)
gs2.fit(x_train3, y_train)
gs2.grid_scores_, gs2.best_params_, gs2.best_score_

log_reg = LogisticRegression(penalty='l1',C=9,random_state= 1)
log_reg.fit(x_train3, y_train)
pred2 = log_reg.predict(x_test3)
#准确率
roc_score2 = roc_auc_score(y_test, pred2)   
acc_score2 = accuracy_score(y_test, pred2) 

#KNN
#网格搜索,'wminkowski',,'seuclidean'
param_test3 = [{'n_neighbors':[x for x in range(3, 15) if x % 2 == 1],'weights':['uniform','distance'],'metric':['euclidean','manhattan','chebyshev']}]
gs3 = GridSearchCV(estimator = neighbors.KNeighborsClassifier(),param_grid = param_test3, scoring='roc_auc',iid=False,cv=5,n_jobs=-1)
gs3.fit(x_train3, y_train)
gs3.grid_scores_, gs3.best_params_, gs3.best_score_ 

knn = neighbors.KNeighborsClassifier(n_neighbors = 11 , weights='distance',metric = 'euclidean')
knn.fit(x_train3, y_train)
pred3 = knn.predict(x_test3)
#准确率
roc_score3 = roc_auc_score(y_test, pred3)   
acc_score3 = accuracy_score(y_test, pred3)

#SVM
#网格搜索'poly','linear', 'rbf', 'sigmoid', 'precomputed'
param_test4 = [{'kernel':['linear', 'rbf', 'sigmoid'],'C':list(range(1,10))}]
gs4 = GridSearchCV(estimator = SVC(random_state=1),param_grid = param_test4, scoring='roc_auc',iid=False,cv=5)
gs4.fit(x_train3, y_train)
gs4.grid_scores_, gs4.best_params_, gs4.best_score_  

svm = SVC(kernel='linear', degree=3, coef0=2, C=1)
svm.fit(x_train3, y_train)
pred4 = svm.predict(x_test3)
#准确率
roc_score4 = roc_auc_score(y_test, pred4)   
acc_score4 = accuracy_score(y_test, pred4)



#19.模型集成 pip install mlxtend
from mlxtend.classifier import StackingClassifier  
sclf = StackingClassifier(classifiers=[gbdt,log_reg,knn], use_probas=True,average_probas=False,meta_classifier= LogisticRegression(penalty='l1',C=4,random_state= 1))
sclf.fit(x_train3, y_train)
pred5 = sclf.predict(x_test3)
#准确率
roc_score5 = roc_auc_score(y_test, pred5)   
acc_score5 = accuracy_score(y_test, pred5)


#20.SelectedGenes下标匹配
u=[]
x_test4=np.array(x_test)
for i in range(x_test3.shape[1]):
    for j in range(x_test4.shape[1]):
        b = (x_test3[:,i] == x_test4[:,j])
        ans = True
        for x in b:
            ans = ans*x
        if ans:
            u.append(j)
            break
        
Genes = list(x_test.columns.values[u])
#与probe ID匹配
probes = pd.read_table(r'C:\Users\\admin\Desktop\\zzy\\lung ML\\TCGA_mRNA_probes.txt',index_col=False,header=0,engine='python',encoding=None,chunksize=None)
id = []
for i in Genes:
    for j in range(probes.shape[0]):
        if i == probes.iloc[j,0]:
            id.append(probes.iloc[j,1])
            break

#21.结果导出
#取出降维后数据
X_reduced = X.loc[:,Genes]
X_reduced.columns = id
X_reduced.to_csv(r'C:\Users\admin\Desktop\zzy\lung ML\X_reduced.txt',sep = '\t')
#预测结果导出(为了之后的预后分析)
pred = Series(sclf.predict(X_reduced))
pred.index = X_reduced.index
pred.to_csv(r'C:\Users\admin\Desktop\zzy\lung ML\pred.txt',sep = '\t')



#作图  
#22.sbs降维分数图
x,y = [],[]
x,y = sbs.plt()
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 4))
plt.ylabel("AUC")
plt.xlabel('number of Genes')
plt.title("Sequential Backward Selection(SBS)")
plt.ylim(ymax=0.990, ymin=0.968)
plt.plot(x[::-1],y[::-1])
plt.savefig(r'C:\\Users\\admin\Desktop\\zzy\\lung ML\\Genes Selection.pdf')
    
#23.箱型图
import seaborn as sns
from pylab import plot,show,savefig,xlim,figure,hold, ylim,legend,boxplot,setp,axes
def setBoxColors(bp):
    setp(bp['boxes'][0], color='green')
    setp(bp['caps'][0], color='green')
    setp(bp['caps'][1], color='green')
    setp(bp['whiskers'][0], color='green')
    setp(bp['whiskers'][1], color='green')
    #setp(bp['fliers'][0], color='blue')
    #setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='green')
    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    #setp(bp['fliers'][2], color='red')
    #setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], color='red')

fig = figure()
ax = axes()
hold(True)
i = 15.5
for j in range(len(Genes)):
    items = [list(X_reduced.iloc[:519,j]),list(X_reduced.iloc[519:,j])]
    bp = boxplot(items, positions = [i-3, i+3], widths = 3,sym='',flierprops = {'marker':'.','markerfacecolor':'black','color':'black'})
    i = i + 20
    setBoxColors(bp)

###下面两句设置 pic 的 X 和 Y 轴的长度，不合适可以修改
xlim(0,254.5)
ylim(-10,22)
plt.xlabel('Genes')
plt.ylabel('Value')
ax.set_xticklabels(id,fontsize=6)
ax.set_xticks([x*20+15.5 for x in range(12)])
hB, = plot([1,1],'g-')
hR, = plot([1,1],'r-')
legend((hB, hR),('LUDA', 'LUSC'),fontsize=7,loc='upper right')
hB.set_visible(False)
hR.set_visible(False)

savefig(r'C:\Users\\admin\Desktop\\zzy\\lung ML\\boxplot_selectedGenes.pdf')

#24.AUC比较图
#全基因预测
path=r'C:\\Users\\admin\Desktop\\zzy\\lung ML\\log_norm.txt'
df=pd.read_table(path,index_col=0,header=0,engine='python',encoding=None,chunksize=None)
X = df.T
x_train,y_train,x_test,y_test = split_train_test(X,Y)

#GBDT
#gbdt=GradientBoostingClassifier(n_estimators=40,learning_rate=0.1,subsample= 0.6,random_state=1)
gbdt.fit(x_train, y_train)
pred10 = gbdt.predict(x_test)
#准确率
roc_score10 = roc_auc_score(y_test, pred10)  
acc_score10 = accuracy_score(y_test, pred10) 


#Logistic回归
#网格搜索
#log_reg = LogisticRegression(penalty='l1',C=1,random_state= 1)
log_reg.fit(x_train, y_train)
pred20 = log_reg.predict(x_test)
#准确率
roc_score20 = roc_auc_score(y_test, pred20)   
acc_score20 = accuracy_score(y_test, pred20) 



#KNN
#网格搜索,'wminkowski',,'seuclidean'
#knn = neighbors.KNeighborsClassifier(n_neighbors = 5 , weights='distance',metric = 'manhattan')
knn.fit(x_train, y_train)
pred30 = knn.predict(x_test)
#准确率
roc_score30 = roc_auc_score(y_test, pred30)   
acc_score30 = accuracy_score(y_test, pred30)

#SVM
#svm = SVC(kernel='rbf', degree=3, coef0=2, C=3)
svm.fit(x_train, y_train)
pred40 = svm.predict(x_test)
#准确率
roc_score40 = roc_auc_score(y_test, pred40)   
acc_score40 = accuracy_score(y_test, pred40)


#模型集成  
sclf = StackingClassifier(classifiers=[gbdt,log_reg,knn], use_probas=True,average_probas=False,meta_classifier= LogisticRegression(penalty='l1',C=4,random_state= 1))
sclf.fit(x_train, y_train)
pred50 = sclf.predict(x_test)
#准确率
roc_score50 = roc_auc_score(y_test, pred50)   
acc_score50 = accuracy_score(y_test, pred50)

AUC_All,AUC_Selected = [],[]
AUC_All.append(roc_score10)
AUC_All.append(roc_score20)
AUC_All.append(roc_score30)
AUC_All.append(roc_score40)
AUC_All.append(roc_score50)
AUC_Selected.append(roc_score1)
AUC_Selected.append(roc_score2)
AUC_Selected.append(roc_score3)
AUC_Selected.append(roc_score4)
AUC_Selected.append(roc_score5)


import matplotlib as mpl
mpl.use('Agg')

font_size = 10 # 字体大小
fig_size = (8, 6) # 图表大小

names = (u'All', u'Selected') 
subjects = ('GBDT','Logistic','KNN','SVM','Ensemble') 
scores = (AUC_All,AUC_Selected) 

# 更新字体大小
mpl.rcParams['font.size'] = font_size
# 更新图表大小
mpl.rcParams['figure.figsize'] = fig_size
# 设置柱形图宽度
bar_width = 0.35

index = np.arange(len(scores[0]))
rects1 = plt.bar(index, scores[0], bar_width, color='orange', label=names[0])
rects2 = plt.bar(index + bar_width+0.05, scores[1], bar_width, color='teal', label=names[1])

# X轴标题
plt.xticks(index +0.025+ bar_width/2, subjects)
# Y轴范围
plt.ylim(ymax=1, ymin=0.75)
# 图表标题
plt.title(u'AUC Comparision')
# 图例显示在图表下方
legend(loc='upper right', fancybox=True)
plt.ylabel("AUC")
plt.xlabel('Models')

# 图表输出到本地
plt.savefig(r'C:\Users\\admin\Desktop\\zzy\\lung ML\\AUC Comparision.pdf')

#25.ROC曲线作图
from sklearn.metrics import roc_curve,auc
kfold=StratifiedKFold(y=y_test,n_folds=5,random_state=1)
sclf.fit(x_train3, y_train)
plt.figure()
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('ROC Curve') 
for i,(train,test) in enumerate(kfold):
    prob=sclf.fit(x_test3[train],y_test.iloc[train]).predict_proba(x_test3[test])
    fpr,tpr,thresholds=roc_curve(y_test.iloc[test],prob[:,1],pos_label=1)
    roc_auc=auc(fpr,tpr)
    plt.plot(fpr,tpr,label='ROC fold %d, AUC=%0.2f'%(i+1,roc_auc))
plt.legend(loc = 'down right')
plt.savefig(r'C:\\Users\\admin\Desktop\\zzy\\lung ML\\ROC曲线.pdf')

#26.混淆矩阵作图
from sklearn.metrics import confusion_matrix 

def plot_confusion_matrix(y_true, y_pred, labels,path,title):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cmap = plt.cm.binary
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    cm = plt.cm.get_cmap('RdYlBu')  
    intFlag = 0 # 标记在图片中对文字是整数型还是浮点型
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        #

        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
                #这里是绘制数字，可以对数字大小和颜色进行修改
            plt.text(x_val, y_val, "%0.2f" % (c,), fontsize=25, va='center', ha='center')        
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cm)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cm)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('Index of True Classes')
    plt.xlabel('Index of Predict Classes')
    plt.savefig(path)
    plt.show()
    
path=r'C:\\Users\\admin\Desktop\\zzy\\lung ML\\confusion_matrix.pdf'
labels = ['LUAD','LUSC']
title = 'Comfusion Matrix'
plot_confusion_matrix(y_test,pred5,labels,path,title)

#27.Nearest Neighbor三维图
from mpl_toolkits.mplot3d import Axes3D  
fig = plt.figure()  
ax = fig.add_subplot(111, projection='3d')  
x_LUAD = X_reduced.iloc[:50,0]
y_LUAD = X_reduced.iloc[:50,1]
z_LUAD = X_reduced.iloc[:50,3]
x_LUSC = X_reduced.iloc[520:570,0]
y_LUSC = X_reduced.iloc[520:570,1]
z_LUSC = X_reduced.iloc[520:570,3]

ax.scatter(x_LUAD, y_LUAD, z_LUAD, marker = 'x', color = 'orange', label='LUAD', s = 30)
ax.scatter(x_LUSC, y_LUSC, z_LUSC, marker = 'v', color = 'teal', label='LUSC', s = 50)

ax.set_xlabel('CERS3')
ax.set_ylabel('TRIM7')
ax.set_zlabel('CALMN3')
plt.legend(loc = 'upper left')
plt.savefig(r'C:\\Users\\admin\Desktop\\zzy\\lung ML\\Nearest Neighbour.pdf')


#28.热力图数据导出,之后使用R语言5.pheatmap
X_reduced.T.to_csv(r'C:\Users\admin\Desktop\zzy\lung ML\heatmap.txt',sep = '\t')












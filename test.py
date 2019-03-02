# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import pickle
from hyperopt import hp
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from train_model import Generate_Model,Best_Model
from select_feature import SelectKBestFeature
from estimator import MyEstimator,EstimatorTpye

"""
hp.choice：类别变量
hp.quniform：离散均匀（整数间隔均匀）
hp.uniform：连续均匀（间隔为一个浮点数）
hp.loguniform：连续对数均匀（对数下均匀分布）
"""

#定义模型选择标准
def AccuracyScore(y_true, y_pred):
    accuracy = accuracy_score(y_true,y_pred);
    return 1-accuracy;


#定义选择特征选择方法
selector = SelectKBestFeature(func=mutual_info_classif,k=10,big_is_better=True,param_dic={'n_neighbors':4,'random_state':13});

"""
设定需要训练的学习模型算法
案例中设定了SVC\RandomForestClassifier\SGDClassifier 三种学习模型
使用MyEstimator类实例化学习模型，传入学习模型的类型和名称
"""
clf = SVC();
svc_estimator = MyEstimator(clf,EstimatorTpye.binary_classic,'SVC');

clf = RandomForestClassifier();
rf_estimator = MyEstimator(clf,EstimatorTpye.binary_classic,'randomForestClassifier');

clf = SGDClassifier();
sgd_estimator = MyEstimator(clf,EstimatorTpye.binary_classic,'sgdclassifier');

myestimators = [svc_estimator,rf_estimator,sgd_estimator];

"""
设定模型的参数空间
更具具体的学习模型，设定各自需要优化的参数的参数区间或者分布
"""

svc_space = {'C':hp.lognormal('C', 0, 1),
             'degree':hp.choice('degree',[3,4,5]),
             'kernel':hp.choice('kernel',['rbf','linear'])};

rf_space = {'n_estimators':hp.choice('n_estimators',[5,10,15,18]),
            'max_depth':hp.choice('max_depth',[3,4,5]),
            'min_samples_leaf':hp.choice('min_samples_leaf',[1,2,3]),
            'min_impurity_split':hp.uniform('min_impurity_split',1e-7,1e-6)}

sgd_space = {'loss':hp.choice('loss',['hinge','log','modified_huber','squared_hinge']),
            'penalty':hp.choice('penalty',['l2','l1']),
            'learning_rate':hp.choice('learning_rate',['constant','optimal','invscaling','adaptive']),
            'eta0':hp.uniform('eta0',0.5,1.0)};

myspaces = [svc_space,rf_space,sgd_space];

#定义需要的特征列
columns = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean',
           'radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se',
           'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'];
#准备数据
data = pd.read_csv('data.csv',encoding='utf-8');
data['y'] = 0;
data['y'].loc[data['diagnosis']=='M'] = 1;
cell_id = data[['id','y']];
data  = data[columns];

"""
#切分数据，分为训练集和测试集
训练集用于学习选择学习模型的最佳参数和学习模型的训练
测试集用于选择出效果最好的模型，其中模型优劣根据AccuracyScore函数决定
"""
X_train,X_test,Y_train,Y_test = train_test_split(data.values,cell_id.values,train_size=0.7);

models = [];
select_indexs = [];
scores = [];

for myestimator,myspace in zip(myestimators,myspaces):

    generate = Generate_Model(myestimator=myestimator,log_path='test_train_model.csv',space=myspace,max_evals=5,selector=selector);
    best_model = Best_Model(generate_model=generate,model_path=myestimator.name + '.pkl',func=AccuracyScore);
    best_model,best_select_index,score = best_model.select_best_model(X_train,X_test,Y_train[:,-1],Y_test[:,-1]);
    models.append(best_model);
    select_indexs.append(best_select_index);
    scores.append(score);

"""
优中选优
#从SVC\RandomForestClassifier\SGDClassifier 三种参数最优的模型中选择效果最好的模型
"""
model_names = [clf.name for clf in myestimators];
scores = np.array(scores);
best_index = np.argmin(scores);
best_model = models[best_index];
best_select_index = select_indexs[best_index];
min_error_score = scores[best_index];

#修改最好模型的持久化文件名
best_model_name = model_names[best_index];
if(os.path.exists(best_model_name+'.pkl')):
    os.rename(best_model_name+'.pkl','bestmodel-'+best_model_name+'.pkl');
else:
    pickle.dump((best_model,best_select_index),open(best_model_name+'.pkl','wb'));

import matplotlib.pyplot as plt;

"""
可视化SVC\RandomForestClassifier\SGDClassifier 三种学习模型在AccuracyScore函数
下的得分
"""
colors = ['blue' for i in range(len(myestimators))];
colors[best_index] = 'red';
plt.title('model comparison');
plt.ylabel('error rate');
plt.xlabel('model name');
plt.ylim(bottom=0, top=1);

plt.bar(x=model_names,height=scores,width=0.5,color=colors);
plt.show();

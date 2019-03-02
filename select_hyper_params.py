# -*- coding: utf-8 -*-

#date 2019-01-21 11:36
#author markli(xlihyperlink

from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from hyperopt import space_eval
import os
import csv
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import time
from estimator import EstimatorTpye
"""
set_params 格式
from sklearn.svm import SVC

params = {'C': [.1, 1, 10]}

for k, v in params.items():
    for val in v:
        clf = SVC().set_params(**{k: val})
        print(clf)
"""


    
class HyperParamsOpt:
    def __init__(self,myestimator,log_path,X,y,space,max_evals):
        """
        estimator 需要优化超参数的算法
        log_path 记录每次迭代的日志文件路径
        X 训练数据
        y 训练数据
        estimator_type 问题类型，regression，binarry_classic,multiclass
        space 超参数取值空间dict
        """
        self.myestimator = myestimator;
        self.log = log_path;
        self.X = X;
        self.Y = y;
        self.space = space;
        self.max_evals = max_evals;
        
        if(not os.path.exists(log_path)):
            of_connection = open(log_path, 'w',encoding='utf-8',newline='');
            writer = csv.writer(of_connection);
            
            writer.writerow(['datetime','loss', 'params', 'estimators', 'train_time']);
            of_connection.close();
    
    
    def write_log(self,datetime,loss, params,estimator_name, run_time):
        """
        datetime 当前日期时间
        loss 需要优化的算法的输出值
        params 算法的超参数
        estimator_name 算法名称
        run_time 本次优化所需的时间
        """
        of_connection = open(self.log, 'a',encoding='utf-8',newline='');
        writer = csv.writer(of_connection);
        writer.writerow([datetime,loss, params, estimator_name, run_time]);
        of_connection.close();
        
    def regression_objective(self,params, n_folds=3):
       '''
       回归问题的参数优化，使用mean_squared_error作为最小化优化依据
       每次选择本次结果中的最坏的情况，最坏的结果改善，则整体得到改善
       params 算法的超参数
       '''
       starttime = time.time();
       scorer = metrics.make_scorer(metrics.mean_squared_error,greater_is_better=False);
       self.myestimator.estimator = self.myestimator.estimator.set_params(**params);
       cv_results = cross_val_score(estimator=self.myestimator.estimator,X=self.X,y=self.Y,scoring=scorer,cv=n_folds);
       endtime = time.time();
       train_time = int(endtime-starttime);
       worse_score = max(cv_results);
       # Dictionary with information for evaluation
       datetime = time.strftime('%Y-%m-%d %H:%M:%S');
       self.write_log(datetime,worse_score,params,self.myestimator.name,train_time)
       return {'loss': worse_score, 'params': params, 'status': STATUS_OK};
   
    def binary_classic_objective(self,params,n_folds=3):
        """
        二分类问题的参数优化，使用roc_auc_score作为最小化优化依据
        每次选择本次结果中的最坏的情况，最坏的结果改善，则整体得到改善
        params 算法的超参数
        """
        starttime = time.time();
        scorer = metrics.make_scorer(metrics.roc_auc_score,greater_is_better=True);
        self.myestimator.estimator = self.myestimator.estimator.set_params(**params);
        cv_results = cross_val_score(estimator=self.myestimator.estimator,X=self.X,y=self.Y,scoring=scorer,cv=n_folds);
        endtime = time.time();
        train_time = int(endtime-starttime)
        worse_score = min(cv_results);
        worse_score = 1 - worse_score;
        datetime = time.strftime('%Y-%m-%d %H:%M:%S');
        self.write_log(datetime,worse_score,params,self.myestimator.name,train_time)
       # Dictionary with information for evaluation
        return {'loss': worse_score, 'params': params, 'status': STATUS_OK};
    
    def multiclass_objective(self,params,n_folds=3):
        """
        多分类问题的参数优化，使用hinge_loss作为最小化优化依据
        每次选择本次结果中的最坏的情况，最坏的结果改善，则整体得到改善
        params 算法的超参数
        """
        starttime = time.time();
        scorer = metrics.make_scorer(metrics.hinge_loss,greater_is_better=False);
        self.myestimator.estimator = self.myestimator.estimator.set_params(**params);
        cv_results = cross_val_score(estimator=self.myestimator.estimator,X=self.X,y=self.Y,scoring=scorer,cv=n_folds);
        endtime = time.time();
        train_time = int(endtime-starttime);
        worse_score = max(cv_results);
        # Dictionary with information for evaluation
        datetime = time.strftime('%Y-%m-%d %H:%M:%S');
        self.write_log(datetime,worse_score,params,self.myestimator.name,train_time)
        return {'loss': worse_score, 'params': params, 'status': STATUS_OK};
    
    def optimizition(self):
        bayes_trials  = Trials();
        if(self.myestimator.estimator_type == EstimatorTpye.binary_classic or self.myestimator.estimator_type == EstimatorTpye.binary_classic.value):
            best = fmin(fn=self.binary_classic_objective,space=self.space,algo=tpe.suggest,max_evals=self.max_evals,trials=bayes_trials);
        elif(self.myestimator.estimator_type == EstimatorTpye.multiclass or self.myestimator.estimator_type == EstimatorTpye.multiclass.value):
            best = fmin(fn=self.multiclass_objective,space=self.space,algo=tpe.suggest,max_evals=self.max_evals,trials=bayes_trials);
        elif(self.myestimator.estimator_type == EstimatorTpye.regression or self.myestimator.estimator_type == EstimatorTpye.regression.value):
            best = fmin(fn=self.regression_objective,space=self.space,algo=tpe.suggest,max_evals=self.max_evals,trials=bayes_trials);
        else:
            print("the type of the estimator is not define");
            return 0;
        #设定算法的最优参数
        params = space_eval(self.space,best);
           
        print("the best params is %s" % params);
        self.myestimator = self.myestimator.set_params(**params);
        
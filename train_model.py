# -*- coding: utf-8 -*-

#date 2019-01-22 13:46
#author markli(xli015)

from select_feature import SelectKBestFeature
from select_hyper_params import HyperParamsOpt
from estimator import EstimatorTpye,MyEstimator
import pickle
import sys

class Generate_Model:
    def __init__(self,myestimator,log_path,space,max_evals,selector):
        """
        myestimator MyEstimator 对象
        log_path 模型寻找最优参数时log文件路径
        space 模型参数可行域
        max_evals 最大的迭代次数
        seletor SelectKBestFeature 对象，用于选择最优特征
        """
        self.myestimator = myestimator;
        self.log_path = log_path;
        #self.feature_select = feature_select_method;
        self.space = space;
        self.max_evals = max_evals;
        self.selector = selector;
    
    def generate(self,X,y):
        #特征选择
        select_index = self.selector.fit(X,y);
        X = X[:,select_index];
        #参数选择
        hyperparam = HyperParamsOpt(self.myestimator,self.log_path,X,y,self.space,self.max_evals);
        hyperparam.optimizition();
        
        #模型训练
        self.myestimator.estimator.fit(X,y);
        #pickle.dump((self.myestimator.estimator,select_index),open(model_path,'wb'));
        return self.myestimator.estimator,select_index;



class Best_Model:
    def __init__(self,generate_model,model_path,func):
        #self.X_train = X_train;
        #self.X_test = X_test;
        #self.Y_train = Y_train;
        #self.Y_test = Y_test;
        self.model_path = model_path;
        self.generate = generate_model;
        self.func = func;
    
    def select_best_model(self,X_train,X_test,Y_train,Y_test):
        """
        model_path 最优模型保存的路径
        func 模型评价函数func(y_true,y_pre) return score；score 越小表示越好
        """
        n,m = X_test.shape;
        if(not callable(self.func)):
            print("the func is not callable");
            return None;
        pre_score = sys.maxsize;
        best_model = None;
        best_select_index = None;
        #遍历特征
        for i in range(1,m+1):
            setattr(self.generate.selector,'k',i);
            estimator,select_index = self.generate.generate(X_train,Y_train);
            if(estimator is None):
                return None;
            y_pre = estimator.predict(X_test[:,select_index]);
            score = self.func(Y_test,y_pre);
            if(score<pre_score):
                pre_score = score;
                best_model = estimator;
                best_select_index = select_index;
        pickle.dump((best_model,best_select_index),open(self.model_path,'wb'));
        return best_model,best_select_index,pre_score;
                
                
            
                
        
        
        
                    
        
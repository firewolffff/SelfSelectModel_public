# -*- coding: utf-8 -*-

#date 2019-01-23 11:12
#author markli(xli015)

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import RFE
from pandas import Series
import numpy as np


class SelectKBestFeature:
    def __init__(self,func,k,big_is_better=False,param_dic={}):
        """
        k 需要选择的特征数量,k==-1表示取所有的特征 k<0 and k>1 表示选择特征的比例
        big_is_better 标记该特征选择方法计算出的指标值是否时越大越好
        param_dic func中的参数
        """
        self.func = func;
        self.params = param_dic;
        self.k = k;
        self.big_is_better = big_is_better;
    
    def fit(self,X,y):
        """
        sklearn 中特征选择方法有的封装在类中例如RFE，有的封装成函数例如mutual_info_regression
        因此在向方法中传递参数时需要分开讨论
       
        """
        n,m = X.shape;
        if(self.k == -1 or self.k == m):
            return list(range(m));
        elif(self.k>0 and self.k<1):
            n_features = int(m*self.k);
        else:
            n_features = self.k;
        
        #func是函数类型
        # 根据sklearn中feature_selection 中方法的返回值看，若返回两个结果则其中一个为score,一个为p，此时选择p作为特征选择标准；
        #若返回数组，则表示指标值，则此时根据返回结果选择特征
        #func 是类对象 必须存在fit(X,y) 和 get_support(indices)方法，参数k此时不起作用
        #根据sklearn 中feature_selection中使用类对象方式选择参数时，会产生两个属性，但属性名不尽相同。它们的共同点是每个类都有一个
        #get_support方法返回选定的feature索引，get_support方法中indices=False默认，返回bool类型数组，指示选定的feature，indices=True，返回选定feature索引号
        #为了实现接口的统一处理，选定get_support(indices=True)，此时k参数将不起作用
        if(type(self.func).__name__ == 'function'):
            self.params['X'] = X;
            self.params['y'] = y;
            result = self.func(**self.params);
            if(isinstance(result,tuple)):
                score = result[-1];
            else:
                score = result;
        elif(hasattr(self.func,'fit') and hasattr(self.func,'get_support')):
            self.func = self.func.set_params(**self.params);
            self.func = self.func.fit(X,y);
            select_index = self.func.get_support(indices=True);
            return list(select_index);
        else:
            print("the func must be a type of function or a class with fit method");
            return list(range(n_features));
        
        
        score = Series(score);
        if(self.big_is_better):
            score = score.sort_values(ascending=False);
        else:
            score = score.sort_values(ascending=True);
        
        return list(score[:n_features].index);
    
            
            
                
                    
                
            
            
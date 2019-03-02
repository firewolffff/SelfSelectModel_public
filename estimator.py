# -*- coding: utf-8 -*-


#date 2019-01-22 13:46
#author markli(xli015)
from enum import Enum, unique

@unique
class EstimatorTpye(Enum):
    regression = 'regression';
    binary_classic = 'binary_classic';
    multiclass = 'multiclass';
    
class MyEstimator:
    def __init__(self,estimator,estimator_type,estimator_name):
        self.estimator = estimator;
        self.name = estimator_name;
        if(type(estimator_type) == str):
            try:
                self.estimator_type = EstimatorTpye[estimator_type];
            except KeyError:
                print('the estimator type is not define,the validate type is regression,binary_classic,multiclass');
                return None;
        elif(type(estimator_type) == EstimatorTpye):
            self.estimator_type = estimator_type;
        else:
            print('the estimate type is not fit.please input str or EstimatorType');
            return None;
        
    def get_estimator(self):
        return self.estimator;
    
    def get_estimator_name(self):
        return self.name;
    
    def set_params(self,**params):
        self.estimator = self.estimator.set_params(**params);
        return self;
    
    def get_params(self):
        return self.estimator.get_params();
        
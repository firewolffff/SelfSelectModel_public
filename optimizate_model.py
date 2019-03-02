# -*- coding: utf-8 -*-

#date 2019-01-24 10:00
#author markli(xli015)

import os
import pickle
import csv
import time

class Application:
    """
    def __init__(self,model,tol,check_point):
        #model 训练好的模型二元组，包括模型和选定的特征索引号(estimator,select_index)
        #tol 允许的误差值
        #check_point 检查点。可以是时间，也可以是数量。若是时间,则格式为 num[D,d,M,m,Y,y] str,D,d 表示天，M,m表示月，Y,y表示年。若是数量，则格式为N int
        if(isinstance(model,str)):
            if(os.path.exists(model)):
                self.estimator,self.select_index = pickle.load(open(model,'rb'));
            else:
                print('the model does not exists');
                return None;
        else:
            self.estimator,self.select_index = model;
        
        self.tol = tol;
        
        self.time_end = sys.maxsize;
        self.end_num = sys.maxsize;
        #时间格式
        if(isinstance(check_point,str)):
            try:
                num = int(check_point[:-1]);
            except ValueError:
                print("please input the right format num[D,d,M,m,Y,y],for example 3d");
                return None;
            unit = check_point[-1];
            time_now = time.time();
            #self.time_now = time_now;
           
            if(unit.lower() == 'd'):  
                self.time_end = time_now + num*24*60*60;
            elif(unit.lower() == 'm'):
                self.time_end = time_now + num*30*24*60*60;
            elif(unit.lower() == 'y'):
                self.time_end = time_now + num*12*30*24*60*60;
            else:
                print("the time format is wrong.the check_point will be setted one mouth");
                self.time_end = time_now + 30*24*60*60;
        #数量格式
        elif(isinstance(check_point,int)):
            self.end_num = check_point;
        else:
            print("the check_point format is not defined");
            return None;
    """
    def __init__(self,model,data_online,columns):
        if(isinstance(model,str)):
            if(os.path.exists(model)):
                self.estimator,self.select_index = pickle.load(open(model,'rb'));
            else:
                print('the model does not exists');
                return None;
        else:
            self.estimator,self.select_index = model;
        self.data_online = data_online;
        file = open(data_online,'w');
        writer = csv.writer(file);
        columns.append('pre_y');
        columns.append('datetime');
        writer.writerow(columns);
        file.close();
        
            
    def predict(self,X):
        y = self.estimator.predict(X[:,self.select_index]);
        file = open(self.data_online,'w');
        writer = csv.writer(file);
        for i in range(X.shape[1]):
            item = list(X[i]);
            item.append(y[i]);
            item.append(time.strftime("%Y-%m-%d %H:%M:%S"));
            writer.writerow(item);
        file.close();
        
            
        
        
            
        
    
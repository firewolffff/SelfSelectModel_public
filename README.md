# selfselectmodel
该框架实现了基于Sklearn的模型自动优化，特征自动选择
test.py提供了框架的使用案例
框架需要传入3个自定义函数:
	1、SelectKBestFeature()类需要传入用于选择特征的自定义函数；
	2、HyperParamsOpt()类需要传入用于训练的模型；
	3、Best_Model()类需要传入用于评价模型的自定义函数;

HyperParamsOpt()类中实例化了三种评价模型参数选择的方法。回归模型评价方法、二分类模型评价方法、多分类模型评价方法。
用户可更具自己实际需求修改模型参数选择的评价方法。

# 功能描述（Function Description）
一个python版本的逐步回归，提供了逐步逻辑回归和逐步线性回归  
在添加特征和删除特征时使用多进程来并行计算来决定每个特征是否应该加入或删除模型  
支持多进程,Windows系统的多进程版本也被支持  
支持的功能点如下：  
1.支持双向逐步回归(Step_Wise)  
2.支持多进程，在每步增加变量或删除变量时，使用多进程来遍历每个候选变量。Windows系统也支持多进程  
3.支持使用者指定的指标来作为变量添加或删除的依据，而不是使用AIC或BIC，在处理不平衡数据时可以让使用者选择衡量不平衡数据的指标  
4.支持使用者指定P-VALUE的阈值，如果超过该阈值，即使指标有提升，也不会被加入到变量中  
5.支持使用者指定VIF的阈值，如果超过该阈值，即使指标有提升，也不会被加入到变量中  
6.支持使用者指定相关系数的阈值，如果超过该阈值，即使指标有提升，也不会被加入到变量中  
7.支持使用者指定回归系数的正负号，在某些业务中，有些特征有明显的业务含义，例如WOE转换后的数据，就会要求回归系数均为正或均为负，加入对系数正负号的限制，如果回归系数不满足符号要求，则当前变量不会被加入到变量中  
8.上述4，5，6，7均在逐步回归中完成，挑选变量的同时校验各类阈值与符号  
9.会给出每一个没有入模变量被剔除的原因，如加入后指标下降，P-VALUE超出指定阈值，正负号与使用者的预期不符等等  
10.支持中英文双语的日志，会将逐步回归中的每一轮迭代的情况记录到中文日志和英文日志中  


A step-wise regression with python.It has step-wise logstic regression and step-wise linear regression.  
It uses multiprocessing when deciding to add or remove features.   
It works with multi-processing.Supporting Windows system multi-processing too.  
The characteristics are listed below:  
1.Supporting forward-backward Step-Wise.  
2.Supporting multi-processing.When adding or removing features,multi-processing is used to traversal all candidate features.  
3.Supporting that user could point the index instead of AIC/BIC for measuring model performance when adding or removing feaures.That is benifit when user\`s data is unbalanced.  
4.Supporting that user could point p-value threshold.If max p-value is more than this threshold,the current features will not be added,although getting a lift on performance of model.  
5.Supporting that user could point VIF threshold.If max VIF is more than this threshold,the current features will not be added,although getting a lift on performance of model.  
6.Supporting that user could point coefficient of correlation threshold.If max coefficient of correlation is more than this threshold,the current features will not be added,although getting a lift on performance of model.  
7.Supporting that user could point sign to coefficients of regression. A part of features have sense in some business like woe transfer which require that all coefficients of regression are postive or negtive.If the signs requirement is not met,the current features will not be added,although getting a lift on performance of model.  
8.[4,5,6,7] above are completed in step-wise procedure.Picking features and verifing those thresholds and signs are simultaneous.  
9.Users will get reasons of which features isn\`t picked up,as performance is fall or p-value is more than threshold or signs is not in accord with user\`s expect and so on after adding this feature.  
10.Supporting the Chinese and English log in whcih user can get record of every iteration.  
 
# 使用说明 （Usage）:
```
import MutliProcessMStepRegression as mpmr
from sklearn.datasets import make_classification,make_regression
import pandas as pd
 
def get_X_y(data_type):
    if data_type == 'logistic':
        #含有信息的变量个数：4个
        #冗余变量个数：2。 冗余变量是有信息变量的线性组合
        #无用变量个数=10-4-2=4。
        
        #number of informative features = 4
        #number of redundant features = 2.redundant feature is linear combinations of the informative features
        #number of useless features = 10-4-2=4
        X, y = make_classification(n_samples=200,n_features=10,n_informative=4,n_redundant=2,shuffle=False,random_state=0,class_sep=2)
        X = pd.DataFrame(X,columns=['informative_1','informative_2','informative_3','informative_4','redundant_1','redundant_2','useless_1','useless_2','useless_3','useless_4']).sample(frac=1)
        y=pd.Series(y).loc[X.index]
        
    if data_type == 'linear':
        # 含有信息的变量个数：6个
        # 特征矩阵的秩：2个（说明含有信息量的6个变量中存在共线性）
        
        # number of informative features = 6
        # matrix rank = 2 (implying collinearity between six informative features)
        X, y = make_regression(n_samples=200,n_features=10,n_informative=6,effective_rank=2,shuffle=False,random_state=0)#
        X = pd.DataFrame(X,columns=['informative_1','informative_2','informative_3','informative_4','informative_5','informative_6','useless_1','useless_2','useless_3','useless_4']).sample(frac=1)
        y=pd.Series(y).loc[X.index]
    return X, y
    
def test_logit(X,y):
  #   从结果可以看出：
  #   1.算法选出了全部有效变量。
  #   2.排除了所有线性组合变量，而且排除的理由是超出VIF或相关系数的设置或系数不显著。
  #   3.排除了所有无效变量，排除的原因是模型性能没有提升或系数不显著
    
  #   As can be seen:
  #   1.All informative features are picked up by this algorithm
  #   2.All linear combinations features are excluded and the reasons are over the max_vif_limit or  max_corr_limit or max_pvalue_limit
  #   3.All useless features are excluded and the reasons are no lift on the perfermance of model or over max_pvalue_limit
    
  #   return
  #    in_vars = ['informative_3', 'informative_4', 'informative_2', 'informative_1']
  #
  #    dr = {'redundant_1': (['模型性能=0.956100,小于等于最终模型的性能=0.956100',
  #   '最大VIF=inf,大于设置的阈值=3.000000',
  #   '最大相关系数=0.925277,大于设置的阈值=0.600000',
  #   '有些系数不显著，P_VALUE大于设置的阈值=0.050000'],
  #  ['the performance index of model=0.956100,less or equals than the performance index of final model=0.956100',
  #   'the max VIF=inf,more than the setting of max_vif_limit=3.000000',
  #   'the max correlation coefficient=0.925277,more than the setting of max_corr_limit=0.600000',
  #   'some coefficients are not significant,P_VALUE is more than the setting of max_pvalue_limit=0.050000']),
  # 'redundant_2': (['模型性能=0.956100,小于等于最终模型的性能=0.956100',
  #   '最大VIF=inf,大于设置的阈值=3.000000',
  #   '最大相关系数=0.676772,大于设置的阈值=0.600000',
  #   '有些系数不显著，P_VALUE大于设置的阈值=0.050000'],
  #  ['the performance index of model=0.956100,less or equals than the performance index of final model=0.956100',
  #   'the max VIF=inf,more than the setting of max_vif_limit=3.000000',
  #   'the max correlation coefficient=0.676772,more than the setting of max_corr_limit=0.600000',
  #   'some coefficients are not significant,P_VALUE is more than the setting of max_pvalue_limit=0.050000']),
  # 'useless_1': (['模型性能=0.955200,小于等于最终模型的性能=0.956100',
  #   '有些系数不显著，P_VALUE大于设置的阈值=0.050000'],
  #  ['the performance index of model=0.955200,less or equals than the performance index of final model=0.956100',
  #   'some coefficients are not significant,P_VALUE is more than the setting of max_pvalue_limit=0.050000']),
  # 'useless_2': (['有些系数不显著，P_VALUE大于设置的阈值=0.050000'],
  #  ['some coefficients are not significant,P_VALUE is more than the setting of max_pvalue_limit=0.050000']),
  # 'useless_3': (['有些系数不显著，P_VALUE大于设置的阈值=0.050000'],
  #  ['some coefficients are not significant,P_VALUE is more than the setting of max_pvalue_limit=0.050000']),
  # 'useless_4': (['模型性能=0.955800,小于等于最终模型的性能=0.956100',
  #   '有些系数不显著，P_VALUE大于设置的阈值=0.050000'],
  #  ['the performance index of model=0.955800,less or equals than the performance index of final model=0.956100',
  #   'some coefficients are not significant,P_VALUE is more than the setting of max_pvalue_limit=0.050000'])}
    lr  =  mpmr.LogisticReg(X,y,measure='roc_auc',iter_num=20,logger_file_EN='c:/temp/mstep_en.log',logger_file_CH='c:/temp/mstep_ch.log')
    in_vars,clf_final,dr = lr.fit()
    return in_vars,clf_final,dr
    
 
def test_linear(X,y):
 #    从结果可以看出：
 #    选出的变量来自有信息量的变量
 #    选出变量的个数等于特征矩阵秩的个数
    
 #    As can be seen:
 #    The picked features is from informative features
 #    The number of picked features equals matrix rank
    
 #    return 
 #    ['informative_2', 'informative_6']
 #    dr = {'informative_1': (['有些系数不显著，P_VALUE大于设置的阈值=0.100000'],
 #  ['some coefficients are not significant,P_VALUE is more than the setting of max_pvalue_limit=0.100000']),
 # 'informative_3': (['有些系数不显著，P_VALUE大于设置的阈值=0.100000'],
 #  ['some coefficients are not significant,P_VALUE is more than the setting of max_pvalue_limit=0.100000']),
 # 'informative_4': (['有些系数不显著，P_VALUE大于设置的阈值=0.100000'],
 #  ['some coefficients are not significant,P_VALUE is more than the setting of max_pvalue_limit=0.100000']),
 # 'informative_5': (['有些系数不显著，P_VALUE大于设置的阈值=0.100000'],
 #  ['some coefficients are not significant,P_VALUE is more than the setting of max_pvalue_limit=0.100000']),
 # 'useless_1': (['有些系数不显著，P_VALUE大于设置的阈值=0.100000'],
 #  ['some coefficients are not significant,P_VALUE is more than the setting of max_pvalue_limit=0.100000']),
 # 'useless_2': (['有些系数不显著，P_VALUE大于设置的阈值=0.100000'],
 #  ['some coefficients are not significant,P_VALUE is more than the setting of max_pvalue_limit=0.100000']),
 # 'useless_3': (['有些系数不显著，P_VALUE大于设置的阈值=0.100000'],
 #  ['some coefficients are not significant,P_VALUE is more than the setting of max_pvalue_limit=0.100000']),
 # 'useless_4': (['有些系数不显著，P_VALUE大于设置的阈值=0.100000'],
 #  ['some coefficients are not significant,P_VALUE is more than the setting of max_pvalue_limit=0.100000'])}
    lr  =  mpmr.LinearReg(X,y,iter_num=20,max_pvalue_limit=0.1,logger_file_EN='c:/temp/mstep_en.log',logger_file_CH='c:/temp/mstep_ch.log')
    in_vars,clf_final,dr = lr.fit()
    return in_vars,clf_final,dr
 
#在windows中必须使用__main__才能使多进程被执行
#In windows,must use __main__ to make multi-processing running
if __name__ == '__main__':    
    X_logit, y_logit = get_X_y('logistic')
    in_vars_logit,clf_final_logit,dr_logit = test_logit(X_logit,y_logit)
    
    X_linear, y_linear = get_X_y('linear')
    in_vars_linear,clf_final_linear,dr_linear = test_linear(X_linear,y_linear)
```

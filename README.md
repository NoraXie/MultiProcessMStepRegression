支持的功能点如下：
1.支持双向逐步回归(Step_Wise)
2.支持多进程，在每步增加变量或删除变量时，使用多进程来遍历每个候选变量。Windows系统也支持多进程。
3.支持使用者指定的指标来作为变量添加或删除的依据，而不是使用AIC或BIC，在处理不平衡数据时可以让使用者选择衡量不平衡数据的指标
4.支持使用者指定P-VALUE的阈值，如果超过该阈值，即使指标有提升，也不会被加入到变量中
5.支持使用者指定VIF的阈值，如果超过该阈值，即使指标有提升，也不会被加入到变量中
6.支持使用者指定相关系数的阈值，如果超过该阈值，即使指标有提升，也不会被加入到变量中
7.支持使用者指定回归系数的正负号，在某些业务中，有些特征有明显的业务含义，例如WOE转换后的数据，就会要求回归系数均为正或均为负，加入对系数正负号的限制，如果回归系数不满足符号要求，则当前变量不会被加入到变量中
8.上述4，5，6，7均在逐步回归中完成，挑选变量的同时校验各类阈值与符号
9.会给出每一个没有入模变量被剔除的原因，如加入后指标下降，P-VALUE超出指定阈值，正负号与使用者的预期不符等等。
10.支持中英文双语的日志，会将逐步回归中的每一轮迭代的情况记录到中文日志和英文日志中

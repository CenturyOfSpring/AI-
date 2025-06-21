本软件包旨在于模仿并重写On E cient and Scalable Computation of the Nonparametric
 Maximum Likelihood Estimator in Mixture Models论文的代码内容！！！

https://github.com/YangjingZhang

本软件包为基于 MATLAB 实现的**混合模型非参数最大似然估计计算工具。该工具采用增广拉格朗日方法与部分期望最大化方法**作为主要数值优化策略，用于高效求解大规模混合模型中的非参数估计问题。

其中，核心求解器包括：

DualALM.m：基于对偶增广拉格朗日算法的主优化模块；

PEM.m：结合局部期望操作的迭代更新方法，用于加速收敛。

为了便于用户理解算法流程及运行方式，程序包提供了三个演示文件，分别为：

test_simulation1.m

test_simulation2.m

test_simulation3.m

这些测试脚本展示了算法在不同模拟数据集下的实际运行效果，可作为参数设定、数据格式与调用方式的参考模板。

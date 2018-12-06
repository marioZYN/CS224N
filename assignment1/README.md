# CS224N assigment 1 

这个作业比我预想的花了更多的时间，断断续续的写了一周。其中Q3的wor2vec loss导数的推导花了很久，代码实现的时候踩了很多numpy纬度和广播机制的坑, 这里记录一下提醒自己。

因为之前我一直都不情愿用(3,)这样的向量，而是使用(3,1)这样纬度的向量，导致这次作业浪费了大量时间来进行纬度的转换，不仅容易出错，而且代码冗余复杂。后来沉下心来看了看numpy的文档，才发觉自己一直使用reshape函数来进行矩阵乘法维度的调整实在是太蠢了。

```python
import numpy as np
A = np.ones((5, 4)) # A is a 5x4 matrix
a1 = A[1] # a1 is array([1,1,1,1]) and its dimension is (4,)
a2 = A[2]
r = np.dot(a1, a2) # we can compute this directly, result is inner product of two vectors
a3 = a1.T # pay attention a3 == a1

```
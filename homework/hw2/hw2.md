## 第二章作业

作业要求：在提供的 target encoding 代码基础上采用 cython 进行加速。

- 为了简化，可以不考虑 y 值和 x 值的特殊情况。
- 具体而言，我们假设 y 值只能取 0，1 两个值，x 值只能取整数，且不存在 x 中某类仅对应一个值的情况。
- 除去实现以外，作业中还应该包括自己实现和原始实现的比较以及自己实现的时间。



本次作业在王然老师提供的target encoding代码上进行修改尝试，以期进一步提升代码运行速度。以下代码为直接使用DataFrame的groupby函数去完成target encoding，该方法效率偏低。

```python
# use DataFrame groupby method
def target_mean_v1(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        groupby_result = data[data.index != i].groupby([x_name], as_index=False).agg(['mean', 'count'])
        result[i] = groupby_result.loc[groupby_result.index == data.loc[i, x_name], (y_name, 'mean')]
    return result
```



第二个方法是对DataFrame逐行处理，将中间结果都保存到dictionary里面，最后再得到结果。计算十万行数据用了3.85秒。

```python
def target_mean_v2(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    value_dict = dict()
    count_dict = dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]] - 1)
    return result
```



以下几种方法都加入使用了cython编写。紧接着下面的这个方法3，基本上相当于将方法2用cython代码重新写一遍，但是后面在实验中会看到，这一改变为程序运行速度带来了很大的提升。十万行数据需要0.035秒，和之前的3.85秒相比是十倍的速度提升。

```cython
# use cython code 
cpdef target_mean_v3(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[double] y = np.asfortranarray(data[y_name], dtype=np.float64)
    cdef np.ndarray[double] x = np.asfortranarray(data[x_name], dtype=np.float64)

    target_mean_v3_impl(result, y, x, nrow)
    return result


cdef void target_mean_v3_impl(double[:] result, double[:] y, double[:] x, const long nrow):
    cdef dict value_dict = dict()
    cdef dict count_dict = dict()
    cdef long i
    for i in range(nrow):
        if x[i] not in value_dict.keys():
            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1
        else:
            value_dict[x[i]] += y[i]
            count_dict[x[i]] += 1
    i = 0
    for i in range(nrow):
        result[i] = (value_dict[x[i]] - y[i]) / (count_dict[x[i]]-1)
```



由于限制了$$x$$ 只能取整数，考虑将其定义为long类型，而不是之前的double类型。结果又比之前快了不少，十万行数据跑了0.014秒。

```cython
# set the dtype of x to be long(np.int_) instead of double(np.float64)
cpdef target_mean_v4(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[double] y = np.asfortranarray(data[y_name], dtype=np.float64)
    cdef np.ndarray[long] x = np.asfortranarray(data[x_name], dtype=np.int_)

    target_mean_v4_impl(result, y, x, nrow)
    return result


cdef void target_mean_v4_impl(double[:] result, double[:] y, long[:] x, const long nrow):
    cdef dict value_dict = dict()
    cdef dict count_dict = dict()
    cdef long i
    for i in range(nrow):
        if x[i] not in value_dict.keys():
            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1
        else:
            value_dict[x[i]] += y[i]
            count_dict[x[i]] += 1
    i = 0
    for i in range(nrow):
        result[i] = (value_dict[x[i]] - y[i]) / (count_dict[x[i]]-1)

```



下面的方法5是在之前的方法4基础上，把计算时loop的方法从range改成了Pyrex的loop形式。结果基本没有得到提升，不排除没有进行正确运用的原因，如果王老师/助教老师看到并愿意给一些comment或者指导的话，我将十分感谢。

```cython
# use Pyrex form of for loop
cpdef target_mean_v5(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[double] y = np.asfortranarray(data[y_name], dtype=np.float64)
    cdef np.ndarray[long] x = np.asfortranarray(data[x_name], dtype=np.int_)

    target_mean_v5_impl(result, y, x, nrow)
    return result


cdef void target_mean_v5_impl(double[:] result, double[:] y, long[:] x, const long nrow):
    cdef dict value_dict = dict()
    cdef dict count_dict = dict()
    cdef long i
    for i from 0 <= i < nrow:
        if x[i] not in value_dict.keys():
            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1
        else:
            value_dict[x[i]] += y[i]
            count_dict[x[i]] += 1
    i = 0
    for i from 0 <= i < nrow:
        result[i] = (value_dict[x[i]] - y[i]) / (count_dict[x[i]]-1)
```



最后的方法6，把存储中间结果的dict换成了unordered_map，结果也是不算意料之外，运行速度进一步大幅提升，十万行仅需0.003秒。

```cython
# use unordered_map instead of dict
cpdef target_mean_v6(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[double] y = np.asfortranarray(data[y_name], dtype=np.float64)
    cdef np.ndarray[long] x = np.asfortranarray(data[x_name], dtype=np.int_)

    target_mean_v6_impl(result, y, x, nrow)
    return result


cdef void target_mean_v6_impl(double[:] result, double[:] y, long[:] x, const long nrow):
    cdef unordered_map[int, double] value_dict
    cdef unordered_map[int, double] count_dict
    cdef long i
    for i in range(nrow):
        got = value_dict.find(x[i])
        if got == value_dict.end():
            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1
        else:
            value_dict[x[i]] += y[i]
            count_dict[x[i]] += 1
    i = 0
    for i in range(nrow):
        result[i] = (value_dict[x[i]] - y[i]) / (count_dict[x[i]]-1)
```

具体运行时间的结果如下：

```python
# 10万行数据的运行时间
run time of target_mean_v2: 3.8457727432250977 seconds
run time of target_mean_v3: 0.034906625747680664 seconds
run time of target_mean_v4: 0.013962268829345703 seconds
run time of target_mean_v5: 0.014960289001464844 seconds
run time of target_mean_v6: 0.0029921531677246094 seconds
```

```python
# 100万行数据的运行时间
run time of target_mean_v2: 79.9535493850708 seconds
run time of target_mean_v3: 0.3515753746032715 seconds
run time of target_mean_v4: 0.14162135124206543 seconds
run time of target_mean_v5: 0.1436154842376709 seconds
run time of target_mean_v6: 0.03390932083129883 seconds
```

我注意到一个点，用到Cython的方法（v3到v6）的运行时间这样看来基本是随着数据规模线性变化；但是纯Python代码的v2方法，十万行需要3.8秒，一百万行运行时间来到了80秒，数据规模对其运行速度的影响程度估计比线性要高。
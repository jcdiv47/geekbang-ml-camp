# distutils: language=c++
# cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
from libcpp.unordered_map cimport unordered_map
from cython.parallel import prange


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
        result[i] = (value_dict[x[i]] - y[i]) / (count_dict[x[i]] - 1)

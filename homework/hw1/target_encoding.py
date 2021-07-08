# coding = 'utf-8'
import numpy as np
import pandas as pd
import time
import tm


# use DataFrame groupby method
def target_mean_v1(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        groupby_result = data[data.index != i].groupby([x_name], as_index=False).agg(['mean', 'count'])
        result[i] = groupby_result.loc[groupby_result.index == data.loc[i, x_name], (y_name, 'mean')]
    return result


# sum by rows
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


def run(func, *args):
    start = time.time()
    res = func(*args)
    end = time.time()
    print("run time of {}: {} seconds".format(func.__name__, end - start))
    return res


def main():
    y = np.random.randint(2, size=(100000, 1))
    x = np.random.randint(10, size=(100000, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])
    run(tm.target_mean_v2, data, 'y', 'x')
    run(tm.target_mean_v3, data, 'y', 'x')
    run(tm.target_mean_v4, data, 'y', 'x')
    run(tm.target_mean_v5, data, 'y', 'x')
    run(tm.target_mean_v6, data, 'y', 'x')
    run(tm.target_mean_v7, data, 'y', 'x')


if __name__ == '__main__':
    main()

from __future__ import division

import numpy as np


def k_scatter(data, scatter_lists):
    '''
    Given a 1D array of values, which we'll call "data", and an ordered set
    of k-sized arrays (which we'll call "scatter_lists") of indices into
    "data", generate an ordered set of arrays, where each array corresponds
    to the "data" values for a particular list of indices from
    "scatter_lists".

    For example (note that here k=2, i.e., the scatter lists are of size 2):

    >>> data = [1, 3, 0, 2, 5, 4]
    >>> scatter_lists = [[0, 3], [1, 4], [0, 5]]
    >>> scattered_data = k_scatter(data, scatter_lists)
    >>> scattered_data
    [[1, 2], [3, 5], [1, 4]]
    '''
    stride = len(scatter_lists[0])
    for i, d in enumerate(scatter_lists[1:]):
        assert(len(d) == stride)
    return [[data[i] for i in scatter_list]
            for scatter_list in scatter_lists]


def k_gather(scattered_data):
    '''
    Given a list of k-sized arrays (potentially the output of k-scatter),
    concatenate them into a single array.

    For example (note that here k=2, i.e., the scatter data arrays are of
    size 2):

    >>> scattered_data = [[1, 2], [3, 5], [1, 4]]
    >>> gathered_data = k_gather(scattered_data)
    >>> gathered_data
    array([1, 2, 3, 5, 1, 4])
    '''
    stride = len(scattered_data[0])
    for i, d in enumerate(scattered_data[1:]):
        assert(len(d) == stride)
    return np.concatenate(scattered_data)


def test_k_scatter_gather():
    a1 = np.arange(10)
    a2 = a1.copy()
    np.random.shuffle(a2)
    stride = a1.size // 2
    scatter_lists = [[list(a2).index(i)
            for i in a1[j * stride:(j + 1) * stride]]
                    for j in range(int(np.ceil(a1.size / stride)))]
    scattered_data = k_scatter(a2, scatter_lists)
    gathered_data = k_gather(scattered_data)
    
    assert((gathered_data == a1).all())


if __name__ == '__main__':
    pass

from __future__ import division

import numpy as np

class ScatterManager(object):
    def __init__(self, data, empty_value=0):
        self.data = data
        self.empty_value = empty_value

    def empty_index(self, index):
        return index < 0

    def scatter_list_index(self, i):
        return i

    def k_scatter(self, scatter_lists):
        '''
        Given a 1D array of values, which we'll call "data", and an ordered set
        of k-sized arrays (which we'll call "scatter_lists") of indices into
        "data", generate an ordered set of arrays, where each array corresponds
        to the "data" values for a particular list of indices from
        "scatter_lists".

        For example (note that here k=2, i.e., the scatter lists are of size 2):

        >>> data = [1, 3, 0, 2, 5, 4]
        >>> scatter_lists = [[0, 3], [1, 4], [0, 5]]
        >>> scatter_manager = ScatterManager(data)
        >>> scattered_data = scatter_manager.k_scatter(scatter_lists)
        >>> scattered_data
        [[1, 2], [3, 5], [1, 4]]
        '''
        stride = len(scatter_lists[0])
        for i, d in enumerate(scatter_lists[1:]):
            assert(len(d) == stride)
        np_data = np.array(self.data)
        scattered_data = [[self.empty_value
                if self.empty_index(index) else self.data[index]
                        for index in scatter_lists[self.scatter_list_index(i)]]
                                for i in range(len(scatter_lists))]
        if isinstance(self.data, np.ndarray):
            return scattered_data
        else:
            return [list(d) for d in scattered_data]


class IndirectScatterManager(ScatterManager):
    '''
    Here, we extend `ScatterManager` to allow the `scatter_lists` to be
    processed in a custom order (instead of in a linear sequence).

    For example, consider the following example.

    >>> data = [1, 3, 0, 2, 5, 4]
    >>> scatter_lists = [[0, 3], [1, 4], [0, 5]]
    >>> scatter_manager = IndirectScatterManager(data)

    Notice that here we pass a custom order (i.e., [1, 2, 0]) to the
    `k_scatter()` method.  This specifies the order in which the scatter
    lists should be processed, and, thus, determines the position that
    the list of `data` values corresponding to the scatter list should
    be written to in the scattered data.  In this case, the zero-th
    scatter list is processed/stored last, as shown below.

    >>> scatter_manager.k_scatter(scatter_lists, [1, 2, 0])
    [[3, 5], [1, 4], [1, 2]]
    '''

    def k_scatter(self, scatter_lists, scatter_index_map=None):
        if scatter_index_map is None:
            self._scatter_index_map = range(len(scatter_lists))
        else:
            self._scatter_index_map = scatter_index_map
        result = super(IndirectScatterManager, self).k_scatter(scatter_lists)
        del self._scatter_index_map
        return result

    def scatter_list_index(self, i):
        return self._scatter_index_map[i]


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

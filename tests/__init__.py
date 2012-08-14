from __future__ import division

import numpy as np

from ..scatter_gather import k_scatter, k_gather


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

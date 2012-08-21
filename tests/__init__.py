from __future__ import division

import numpy as np

from ..scatter_gather import ScatterManager, IndirectScatterManager, k_gather
from ..scatter_gather.cuda import scatter_gather


def all_close(np_array1, np_array2, labels=None):
    if labels is None:
        labels = ['Array %d' % i for i in range(2)]
    else:
        labels = labels[:]
    labels.append('Diff')

    label_lengths = [len(l) for l in labels]
    max_label_length = max(label_lengths)


    message_template = '''\
The following indexes have different values: %%s
    %%%ds = %%s
    %%%ds = %%s
    %%%ds = %%s
    ''' % (max_label_length, max_label_length, max_label_length)

    if not np.allclose(np_array1, np_array2):
        conflicts = np.where(np_array1 != np_array2)

        message = message_template % (conflicts,
            labels[0], list(np_array1[conflicts]),
            labels[1], list(np_array2[conflicts]),
            labels[2],
            list(np.absolute(np_array2[conflicts] - np_array1[conflicts]
                    ) / np.maximum(np_array2[conflicts],
                        np_array1[conflicts]) * 100))
        raise ValueError, message


def test_k_scatter_gather():
    a1 = np.arange(10)
    a2 = a1.copy()
    np.random.shuffle(a2)
    stride = a1.size // 2
    scatter_lists = [[list(a2).index(i)
            for i in a1[j * stride:(j + 1) * stride]]
                    for j in range(int(np.ceil(a1.size / stride)))]
    scatter_manager = ScatterManager(a2)
    scattered_data = scatter_manager.k_scatter(scatter_lists)
    gathered_data = k_gather(scattered_data)
    
    assert((gathered_data == a1).all())


def test_k_scatter_cuda():
    data = np.arange(29, dtype=np.int32)[::-1].copy()
    data_count = np.int32(len(data))
    rng = np.random.RandomState()
    rng.seed(0)
    for two_power in range(13):
        scatter_list_count = 1 << two_power
        scatter_lists = [rng.randint(data_count, size=3) for i in range(scatter_list_count)]
        scatter_lists[0][-1] = -2
        scatter_lists[-1][-1] = -2

        scatter_manager = IndirectScatterManager(data)
        scatter_list_order = np.arange(len(scatter_lists), dtype=np.uint32)
        rng.shuffle(scatter_list_order)
        gathered_data_cpu = k_gather(scatter_manager.k_scatter(scatter_lists,
                scatter_list_order))

        gathered_data_cuda = scatter_gather(data, scatter_lists, scatter_list_order)
                #block_count=?, thread_count=256)

        to_print = (('data_id', np.arange(len(data))),
                ('data', data), ('scatter_lists', scatter_lists),
                ('scatter_list_order', scatter_list_order),
                ('gathered_data (CPU)', gathered_data_cpu),
                ('gathered_data (CUDA)', gathered_data_cuda))
        max_label_length = max(len(label) for label, d in to_print)
        format_str = '%%%ds' % (max_label_length + 1)

        #for label, data_ in to_print:
            #print format_str % label, data_

        #try:
        all_close(gathered_data_cpu, gathered_data_cuda, ['CPU', 'CUDA'])
        #except:
            #import pudb; pudb.set_trace()

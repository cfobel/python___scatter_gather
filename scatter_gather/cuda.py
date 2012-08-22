from __future__ import division

import numpy as np

from cuda_common import get_cuda_function, log2ceil, cuda


def scatter_gather(in_data, scatter_lists, scatter_list_order=None, dtype=None,
        thread_count=None, block_count=None):
    if dtype is None:
        dtype = in_data.dtype
    test = get_cuda_function('scatter_gather.cu', 'k_scatter', dtype)
    data = np.array(in_data, dtype=dtype)
    data_count = np.int32(len(data))
    k = np.int32(len(scatter_lists[0]))
    scatter_count = np.int32(len(scatter_lists))
    scatter_lists = np.concatenate(scatter_lists).astype(np.int32)
    gathered_data = np.empty_like(scatter_lists).astype(dtype)

    default_thread_count = 1 << log2ceil(test.get_attribute(
            cuda.function_attribute.MAX_THREADS_PER_BLOCK))

    if thread_count is None:
        thread_count = int(min(scatter_count, default_thread_count))
    if block_count is None:
        block_count = int(np.ceil(scatter_count / thread_count))

    block = (thread_count, 1, 1)
    grid = (block_count, 1, 1)

    # The number of scatter lists to be processed by all thread blocks
    # (except the first, in the case where scatter_count does not divide
    # evenly by block_count)
    common_scatter_count = scatter_count // block_count

    # If scatter_count does not divide evenly by block_count, compute
    # how many extra elements must be processed by the first thread block
    odd_scatter_count = scatter_count % block_count

    shared = int(np.ceil((common_scatter_count + odd_scatter_count) * k)
            ) * dtype.type(0).itemsize

    print 'thread_count: %d' % thread_count
    print 'block_count: %d' % block_count
    print 'shared mem/block: %d' % shared

    if scatter_list_order is None:
        scatter_list_order = np.arange(len(scatter_lists), dtype=np.uint32)

    test(k, data_count, cuda.InOut(data), scatter_count, cuda.In(scatter_lists),
            cuda.In(scatter_list_order), cuda.Out(gathered_data),
            block=block, grid=grid, shared=shared)

    return gathered_data

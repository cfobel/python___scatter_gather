from __future__ import division

from jinja2 import Template, FileSystemLoader, Environment
from path import path
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


def package_root():
    '''
    Return absolute path to scatter_gather package root directory.
    '''
    try:
        script = path(__file__)
    except NameError:
        script = path(sys.argv[0])
    return script.parent.abspath()


def get_include_root():
    return package_root().joinpath('pycuda_include')


def get_template_root():
    return package_root().joinpath('pycuda_templates')


def get_template_loader():
    return FileSystemLoader(get_template_root())


jinja_env = Environment(loader=get_template_loader())


def log2ceil(x):
    return int(np.ceil(np.log2(x)))


def scatter_gather(in_data, dtype=None, thread_count=None):
    code_template = jinja_env.get_template('scatter_gather.cu')
    mod = SourceModule(code_template.render(), no_extern_c=True,
            options=['-I%s' % get_include_root()], keep=True)

    if dtype is None:
        dtype = in_data.dtype
    assert(dtype in [np.int32, np.float32])

    dtype_map = { np.dtype('float32'): 'float', np.dtype('int32'): 'int'}

    try:
        func_name = 'k_scatter_%s' % dtype_map[dtype]
        test = mod.get_function(func_name)
    except cuda.LogicError:
        print dtype, func_name
        raise

    data = np.array(in_data, dtype=dtype)

    shared = len(data) * dtype.itemsize

    default_thread_count = log2ceil(test.get_attribute(
            cuda.function_attribute.MAX_THREADS_PER_BLOCK))

    if thread_count is None:
        thread_count = min(len(data), 1 << default_thread_count)
    print 'thread_count: %d' % thread_count

    block_count = max(1, log2ceil(len(data) / thread_count))
    print 'block_count: %d' % block_count
    
    block = (thread_count, 1, 1)
    grid = (block_count, 1, 1)

    test(np.int32(len(data)), cuda.InOut(data),
            block=block, grid=grid, shared=shared)

    return data

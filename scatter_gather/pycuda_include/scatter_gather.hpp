#ifndef ___SCATTER_GATHER__HPP___
#define ___SCATTER_GATHER__HPP___


namespace scatter_gather {
    #include <stdio.h>
    #include <stdint.h>
    #include <math.h>

    template <class T>
    __device__ void dump_data(int size, volatile T *data) {
        syncthreads();
        if(threadIdx.x == 0) {
            printf("[");
            for(int i = 0; i < size; i++) {
                printf("%d, ", data[i]);
            }
            printf("]\n");
        }
    }


    template <class T>
    __device__ T greatest_power_of_two_less_than(T n) {
        T k = 1;
        while(k < n) {
            k = k << 1;
        }
        return k >> 1;
    }


    template <class T, int _empty_index=-1>
    class ScatterManager {
    public:
        int _data_count;
        volatile T *_data;
        T _empty_value;

        __device__ ScatterManager(int data_count, volatile T *data, T empty_value)
                : _data_count(data_count), _data(data), _empty_value(empty_value) {
        }

        /*
        * Overload `_empty_value` to zero.  Note that this will cause a
        * compilation error if 0 cannot be implicitly casted to type
        * `T`.
        */
        __device__ ScatterManager(int data_count, volatile T *data) : _data_count(
                data_count), _data(data), _empty_value(0) {}

        /*
        * Interpret entry `e` in scatter_lists[] with the value `_empty_index`
        * as an indication that the corresponding entry in the `block_data` array
        * should be set to the value `_empty_value`.
        *
        * e.g., Consider the following:
        *
        *         k = 2
        *         data = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        *         scatter_lists = [[4, -1], [3, 7], [-1, 2]]
        *         scatter_count = 3
        *
        *     Note that some of the entries in `scatter_lists` have a value of -1.
        *     This is the default `_empty_index` value.  Therefore, in the
        *     `block_data` array, any entries corresponding to these -1 indices
        *     should be set to `_empty_value` (in this case `_empty_value` is zero).
        *     The resulting `block_data` array is then:
        *
        *         block_data = [7, 0, 8, 4, 0, 9]
        */
        __device__ void scatter(int k, int scatter_count,
                int32_t *scatter_lists, volatile T *block_data) {
#ifdef DEBUG_SCATTER_GATHER
            if(threadIdx.x == 0 && blockIdx.x == 0) {
                printf("k_scatter(k=%d, scatter_count=%d)\n", k, scatter_count);
            }
#endif
            int passes = ceil((float)scatter_count / blockDim.x);
            for(int j = 0; j < passes; j++) {
                int i = j * blockDim.x + threadIdx.x;
                if(i < scatter_count) {
                    for(int scatter_index = 0; scatter_index < k; scatter_index++) {
                        int index = scatter_lists[i * k + scatter_index];
                        if(index == _empty_index) {
                            block_data[threadIdx.x * k + scatter_index] = _empty_value;
                        } else {
                            T value = _data[index];
                            block_data[threadIdx.x * k + scatter_index] = value;

#ifdef DEBUG_SCATTER_GATHER
                            printf("{'block_id': %d, 'thread_id': %d, 'pass': %d, 'element_id': %d, 'value': %d}\n",
                                    blockIdx.x, threadIdx.x, j, i, value);
                            if(index >= _data_count) {
                                printf("Out of bounds: %d/%d\n", index, _data_count);
                            }
#endif
                        }
                    }
                }
            }
        }
    };


    template <class T>
    __device__ void k_gather(int k, int scatter_count,
            volatile T *block_data, volatile T *gathered_data) {
#ifdef DEBUG_SCATTER_GATHER
        if(threadIdx.x == 0 && blockIdx.x == 0) {
            printf("k_gather(k=%d, scatter_count=%d)\n", k, scatter_count);
        }
#endif
        int passes = ceil((float)scatter_count / blockDim.x);
        for(int j = 0; j < passes; j++) {
            int i = j * blockDim.x + threadIdx.x;
            if(i < scatter_count) {
                for(int scatter_index = 0; scatter_index < k; scatter_index++) {
                    T value = block_data[threadIdx.x * k + scatter_index];
                    gathered_data[i * k + scatter_index] = value;
                }
            }
        }
    }
}

#endif

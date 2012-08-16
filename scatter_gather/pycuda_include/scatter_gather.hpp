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


    template <class T>
    class ScatterManager {
    public:
        int _data_count;
        int _scatter_count;
        volatile T *_data;
        T _empty_value;

        __device__ ScatterManager(int data_count, volatile T *data, T empty_value)
                : _data_count(data_count), _scatter_count(0), _data(data),
                        _empty_value(empty_value) {}

        /*
        * Overload `_empty_value` to zero.  Note that this will cause a
        * compilation error if 0 cannot be implicitly casted to type
        * `T`.
        */
        __device__ ScatterManager(int data_count, volatile T *data) : _data_count(
                data_count), _scatter_count(0), _data(data), _empty_value(0) {}

        /*
        * Interpret entry `e` in scatter_lists[] where `empty_index()`
        * evaluates to `false` as an indication that the corresponding
        * entry in the `block_data` array should be set to the value
        * `_empty_value`.
        *
        * e.g., Consider the following:
        *
        *         k = 2
        *         data = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        *         scatter_lists = [[4, -1], [3, 7], [-1, 2]]
        *         scatter_count = 3
        *
        *     Note that some of the entries in `scatter_lists` have a
        *     value of -1.  This causes `empty_index()` to return
        *     `false` by default.  Therefore, in the `block_data` array,
        *     any entries corresponding to these -1 indices should be
        *     set to `_empty_value` (in this case `_empty_value` is
        *     zero).  The resulting `block_data` array is then:
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
            _scatter_count = scatter_count;
            int passes = ceil((float)_scatter_count / blockDim.x);
            for(int j = 0; j < passes; j++) {
                bool thread_active;
                int scatter_list_index = local_scatter_list_index(j,
                        thread_active);
                if(thread_active) {
                    int block_data_base_index = local_block_data_base_index(j) * k;
                    for(int scatter_index = 0; scatter_index < k; scatter_index++) {
                        int index = scatter_lists[scatter_list_index * k + scatter_index];
                        if(empty_index(index)) {
                            block_data[block_data_base_index + scatter_index] = _empty_value;
                        } else {
                            T value = _data[index];
                            block_data[block_data_base_index + scatter_index] = value;

#ifdef DEBUG_SCATTER_GATHER
                            printf("{'block_id': %2d, 'thread_id': %2d, "
                                    "'pass': %2d, 'scatter_list_index': %2d, "
                                    "'block_data_base_index': %2d, "
                                    "'block_data_index': %2d, "
                                    "'value': %2d, 'data_index': %2d}\n",
                                            blockIdx.x, threadIdx.x,
                                            j, scatter_list_index, block_data_base_index,
                                            block_data_base_index + scatter_index,
                                            value, index);
                            if(index >= _data_count) {
                                printf("Out of bounds: %d/%d\n", index, _data_count);
                            }
#endif
                        }
                    }
                }
            }
            syncthreads();
        }

        __device__ virtual int local_block_data_base_index(int pass_index) {
            return pass_index * blockDim.x + threadIdx.x;
        }

        /*
         * Given the index of the current pass of execution, return the
         * `scatter_lists` index corresponding to the scatter list array to be
         * processed by the current CUDA thread.
         */
        __device__ virtual int local_scatter_list_index(int pass_index,
                bool &thread_active) {
            int result = local_block_data_base_index(pass_index);
            thread_active = (result < this->_scatter_count);
            return result;
        }

        __device__ virtual bool empty_index(int data_index) {
            return data_index < 0;
        }
    };


    template <class T>
    class IndirectScatterManager : public ScatterManager<T> {
    public:
        uint32_t *_scatter_list_order;

        __device__ IndirectScatterManager(int data_count, volatile T *data, T empty_value)
                : ScatterManager<T>(data_count, data, empty_value), _scatter_list_order(NULL) {}

        /*
        * Overload `_empty_value` to zero.  Note that this will cause a
        * compilation error if 0 cannot be implicitly casted to type
        * `T`.
        */
        __device__ IndirectScatterManager(int data_count, volatile T *data)
                : ScatterManager<T>(data_count, data), _scatter_list_order(NULL) {}

        __device__ void scatter(int k, int scatter_count,
                int32_t *scatter_lists, uint32_t *scatter_list_order,
                        volatile T *block_data) {
            _scatter_list_order = scatter_list_order;
            this->_scatter_count = scatter_count;
            ScatterManager<T>::scatter(k, this->_scatter_count, scatter_lists,
                    block_data);
        }

        __device__ virtual int local_scatter_list_index(int pass_index,
                bool &thread_active) {
            int result = ScatterManager<T>::local_scatter_list_index(
                    pass_index, thread_active);
            if(thread_active) {
                result = _scatter_list_order[result];
            }
            return result;
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
                    T value = block_data[i * k + scatter_index];
                    gathered_data[i * k + scatter_index] = value;
                }
            }
        }
    }
}

#endif

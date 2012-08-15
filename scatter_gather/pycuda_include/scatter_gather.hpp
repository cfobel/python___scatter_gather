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


    /*
     * Interpret entry `e` in scatter_lists[] with the value `empty_index`
     * as an indication that the corresponding entry in the `block_data` array
     * should be set to the value `empty_value`.
     *
     * e.g., Consider the following:
     *
     *         k = 2
     *         data = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
     *         scatter_lists = [[4, -1], [3, 7], [-1, 2]]
     *         scatter_count = 3
     *
     *     Note that some of the entries in `scatter_lists` have a value of -1.
     *     This is the default `empty_index` value.  Therefore, in the
     *     `block_data` array, any entries corresponding to these -1 indices
     *     should be set to `empty_value` (in this case `empty_value` is zero).
     *     The resulting `block_data` array is then:
     *
     *         block_data = [7, 0, 8, 4, 0, 9]
     */
    template <class T, int empty_index=-1>
    __device__ void k_scatter(int k, int data_count, volatile T *data,
            int scatter_count, int32_t *scatter_lists, volatile T *block_data,
                    T empty_value) {
        if(threadIdx.x == 0 && blockIdx.x == 0) {
            printf("k_scatter(k=%d, scatter_count=%d)\n", k, scatter_count);
        }
        int passes = ceil((float)scatter_count / blockDim.x);
        for(int j = 0; j < passes; j++) {
            int i = j * blockDim.x + threadIdx.x;
            if(i < scatter_count) {
                for(int scatter_index = 0; scatter_index < k; scatter_index++) {
                    int index = scatter_lists[i * k + scatter_index];
                    if(index == empty_index) {
                        block_data[threadIdx.x * k + scatter_index] = empty_value;
                    } else {
                        T value = data[index];
                        block_data[threadIdx.x * k + scatter_index] = value;

                        printf("{'block_id': %d, 'thread_id': %d, 'pass': %d, 'element_id': %d, 'value': %d}\n",
                                blockIdx.x, threadIdx.x, j, i, value);
                        if(index >= data_count) {
                            printf("Out of bounds: %d/%d\n", index, data_count);
                        }
                    }
                }
            }
        }
    }


    /*
     * Overload k_scatter function to default the `empty_value` to zero.  Note
     * that this will cause a compilation error if 0 cannot be implicitly
     * casted to type `T`.
     */
    template <class T, int empty_index=-1>
    __device__ void k_scatter(int k, int data_count, volatile T *data,
            int scatter_count, int32_t *scatter_lists, volatile T *block_data) {
        k_scatter<T, empty_index>(k, data_count, data, scatter_count, scatter_lists, block_data, 0);
    }
}

#endif

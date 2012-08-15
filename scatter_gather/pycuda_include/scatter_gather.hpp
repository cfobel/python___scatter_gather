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
    __device__ void k_scatter(int size, int k, volatile T *data,
            int scatter_count, int32_t *scatter_lists) {
        if(threadIdx.x == 0 && blockIdx.x == 0) {
            printf("k_scatter(size=%d, k=%d, scatter_count=%d)\n",
                    size, k, scatter_count);
        }
        int passes = ceil((float)size / blockDim.x);
        for(int j = 0; j < passes; j++) {
            int i = j * blockDim.x + threadIdx.x;
            for(int scatter_index = 0; scatter_index < k; scatter_index++) {
                int index = scatter_lists[i * k + scatter_index];
                T value = data[index];

                if(i < size) {
                    printf("{'block_id': %d, 'thread_id': %d, 'pass': %d, 'element_id': %d, 'value': %d}\n",
                            blockIdx.x, threadIdx.x, j, i, value);
                }
            }
        }
    }
}

#endif

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
    __device__ void k_scatter(int size, volatile T *data) {
        int passes = ceil((float)size / blockDim.x);
        for(int k = 0; k < passes; k++) {
            int i = k * blockDim.x + threadIdx.x;

            if(i < size) {
                printf("{'block_id': %d, 'thread_id': %d, 'pass': %d, 'element_id': %d}\n",
                        blockIdx.x, threadIdx.x, k, i);
            }
        }
    }
}

#endif

#include "scatter_gather.hpp"

{% if not c_types -%}
{%- set c_types=["float", "int"] -%}
{%- endif -%}

extern __shared__ float shared_data[];

{% for c_type in c_types -%}
extern "C" __global__ void k_scatter_{{ c_type }}(int size, {{ c_type }} *data) {
    #if 0
    {{ c_type }} *sh_data = ({{ c_type }} *)&shared_data[0];

    for(int i = threadIdx.x; i < size; i += blockDim.x) {
        if(i < size) {
            sh_data[i] = data[i];
        }
    }
    #endif

    scatter_gather::k_scatter<{{ c_type }}>(size, data);

    #if 0
    for(int i = threadIdx.x; i < size; i += blockDim.x) {
        if(i < size) {
            data[i] = sh_data[i];
        }
    }
    #endif
}
{% endfor %}

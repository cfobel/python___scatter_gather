#include "scatter_gather.hpp"

{% if not c_types -%}
{%- set c_types=["float", "int"] -%}
{%- endif -%}

extern __shared__ float shared_data[];

{% for c_type in c_types -%}
extern "C" __global__ void k_scatter_{{ c_type }}(int k, int data_count,
        {{ c_type }} *data, int scatter_count, int32_t *scatter_lists,
        {{ c_type }} *gathered_data) {
    #if 0
    {{ c_type }} *sh_data = ({{ c_type }} *)&shared_data[0];

    for(int i = threadIdx.x; i < data_count; i += blockDim.x) {
        if(i < data_count) {
            sh_data[i] = data[i];
        }
    }
    #endif
    {{ c_type }} *block_data = ({{ c_type }} *)&shared_data[0];

    scatter_gather::ScatterManager<{{ c_type }}> scatter_manager(data_count, data);
    scatter_manager.scatter(k, scatter_count, scatter_lists, &block_data[0]);
    //scatter_gather::dump_data(scatter_count * k, &block_data[0]);
    scatter_gather::k_gather<{{ c_type }}>(k, scatter_count, &block_data[0], gathered_data);

    #if 0
    for(int i = threadIdx.x; i < data_count; i += blockDim.x) {
        if(i < data_count) {
            data[i] = sh_data[i];
        }
    }
    #endif
}
{% endfor %}

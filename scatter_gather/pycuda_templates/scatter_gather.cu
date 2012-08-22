#include <stdint.h>
#include "scatter_gather.hpp"

{% if not c_types -%}
{%- set c_types=["float", "int"] -%}
{%- endif -%}

extern __shared__ float shared_data[];

{% for c_type in c_types -%}
extern "C" __global__ void k_scatter_{{ c_type }}(int k, int data_count,
        {{ c_type }} *data, int scatter_count, int32_t *scatter_lists,
        uint32_t *scatter_list_order, {{ c_type }} *gathered_data) {
    scatter_gather::k_scatter_global<{{ c_type }}>(k, data_count, data,
            scatter_count, scatter_lists, scatter_list_order,
                    ({{ c_type }}*)&shared_data[0], gathered_data);
}
{% endfor %}

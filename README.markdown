Scatter
-------

Given a 1D array of values, which we'll call `data`, and an ordered set
of k-sized arrays (which we'll call `scatter_lists`) of indices into
`data`, generate an ordered set of arrays, where each array corresponds
to the `data` values for a particular list of indices from
`scatter_lists`.

For example (note that here k=2, i.e., the scatter lists are of size 2):

    data = [1, 3, 0, 2, 5, 4]
    scatter_lists = [[0, 3], [1, 4], [0, 5]]
    scattered_data = [[1, 2], [3, 5], [1, 4]]


Gather
------

Given a list of k-sized arrays (potentially the output of k-scatter),
concatenate them into a single array.

For example (note that here k=2, i.e., the scatter data arrays are of
size 2):

    scattered_data = [[1, 2], [3, 5], [1, 4]]
    gathered_data = [1, 2, 3, 5, 1, 4]

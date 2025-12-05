from functools import reduce
from operator import mul


def flatten_right_from(x, dim):
    size = list(x.size())
    assert -len(size) < dim < len(size), "'dim' must be between ({}, {}). " \
        "Found {}!".format(-len(size), len(size), dim)

    if dim < 0:
        dim = len(size) + dim

    remaining_dims = size[:dim]
    flatten_dim = reduce(mul, size[dim:])
    new_size = remaining_dims + [flatten_dim]
    return x.view(new_size)


def get_reduction_params(block_size, input_size):
    """Given block_size and input size find the parameters for reduction:

    Output:
        shape_for_reduction: the shape we use to `view` input to prepare it for reduction
        reduction_dims: the dims we'll do reduction over

    Example::
        Input:
          block_size: (3, 3, 2, 10)
          input_size: (3, 3, 10, 10)

        Output:
          shape_for_reduction: (3, 3, 5, 2, 10)
          reduction_dim: [0, 1, 3, 4]
    """
    assert len(block_size) == len(input_size)
    shape_for_reduction = []
    reduction_dims = []
    cur_dim = 0
    for i in range(len(block_size)):
        if block_size[i] != input_size[i] and block_size[i] > 1:
            assert input_size[i] % block_size[i] == 0, f"Expecting input size at {i} dimension: {input_size[i]} to be divisible by block_size at {i} dimension: {block_size[i]}"
            shape_for_reduction.append(input_size[i] // block_size[i])
            shape_for_reduction.append(block_size[i])
            # reduce over the block_size[i] dim
            reduction_dims.append(cur_dim + 1)
            cur_dim += 2
        else:
            # block_size[i] == input_size[i] or block_size[i] == 1
            shape_for_reduction.append(input_size[i])
            # we only need to reduce over the dimension if block_size is greater than 1
            # otherwise it's already the same as reduced dimension
            if block_size[i] != 1:
                reduction_dims.append(cur_dim)
            cur_dim += 1
    return shape_for_reduction, reduction_dims
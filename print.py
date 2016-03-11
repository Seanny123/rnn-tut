def update_state(current_input, previous_state, wx, wRec):
    return current_input * wx + previous_state * wRec

def forward_states(input_X, wx, wRec):
    states = np.zeros((input_X.shape[0], input_X.shape[1]+1))
    for in_x in range(0, input_X.shape[1]):
        states[:, in_x+1] = update_state(input_X[:, in_x], states[:, in_x], wx, wRec)
    return states

def backward_gradient(input_X, states, grad_out, wRec):
    grad_over_time = np.zeros((input_X.shape[0], input_X.shape[1]+1))
    grad_over_time[:, -1] = grad_out
    wx_grad = 0
    wRec_grad = 0
    # go backwards
    for k in range(X.shape[1], 0, -1):
        # scale the gradient at each level by the previous input
        wx_grad += np.sum(grad_over_time[:, k] * input_X[:, k-1])
        # scale the gradient at each level by the previous state
        wRec_grad += np.sum(grad_over_time[:, k] * states[:, k-1])
        # only the recurrent connection carries over through time
        grad_over_time[:, k-1] = grad_over_time[:, k] * wRec
    return (wx_grad, wRec_grad), grad_over_time

##### Lasgna #####

# two input layers (input and mask)
l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, 2))
l_mask = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))

# two propagating layers
l_forward = lasagne.layers.RecurrentLayer(
    l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
    W_in_to_hid=init_W(),
    W_hid_to_hid=init_W(),
    nonlinearity=non_lin)
l_backward = lasagne.layers.RecurrentLayer(
    l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
    W_in_to_hid=init_W(),
    W_hid_to_hid=init_W(),
    nonlinearity=non_lin,
    backwards=True)

# one concatenating layer
l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])

# one dense layer for the output
l_out = lasagne.layers.DenseLayer(l_concat, num_units=1, nonlinearity=non_lin)


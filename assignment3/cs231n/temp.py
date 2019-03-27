from __future__ import print_function, division
from builtins import range
import numpy as np


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    H = prev_h.shape[1]
    A = x.dot(Wx) + prev_h.dot(Wh) + b #(N, 4H)
    i = sigmoid(A[:, :H])
    f = sigmoid(A[:, H:2*H])
    o = sigmoid(A[:, 2*H:3*H])
    g = np.tanh(A[:, 3*H:])
    next_c = f*prev_c + i*g
    next_h = o*np.tanh(next_c)
    cache = x, Wx, prev_h, Wh, prev_c, i, f, o, g, next_c, A
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    x, Wx, prev_h, Wh, prev_c, i, f, o, g, next_c, A = cache
    H = prev_h.shape[1]
    do = dnext_h * np.tanh(next_c)
    dother = dnext_h * o
    dnext_c_sum = (1 - np.power(np.tanh(next_c), 2))*dother + dnext_c
    dprev_c = dnext_c_sum * f
    df = dnext_c_sum * prev_c
    di = dnext_c_sum * g
    dg = dnext_c_sum * i
    di = sigmoid(A[:, :H])*(1-sigmoid(A[:, :H])) * di
    df = sigmoid(A[:, H:2*H])*(1-sigmoid(A[:, H:2*H])) * df
    do = sigmoid(A[:, 2*H:3*H])*(1-sigmoid(A[:, 2*H:3*H])) * do
    dg = (1-np.power(np.tanh(A[:, 3*H:]), 2)) * dg
    dA = np.hstack((di, df, do, dg))
    dWx = np.dot(x.T, dA)
    dx = np.dot(dA, Wx.T)
    dWh = np.dot(prev_h.T, dA)
    dprev_h = np.dot(dA, Wh.T)
    db = np.sum(dA, axis=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db

def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    N, T, D = x.shape
    H = h0.shape[1]
    h_t = np.zeros((T, N, H))
    h = np.zeros((N, T, H))
    c = np.zeros((T, N, H))
    cache = []
    for t in range(T):
        xn = x[:, t, :].reshape((N,D))
        if t==0:
            prev_h, prev_c = h0, c[t]
        else:
            prev_h, prev_c = h_t[t-1], c[t-1]
        h_t[t], c[t], new_cache = lstm_step_forward(xn, prev_h, prev_c, Wx, Wh, b)
        cache.append(new_cache)
    #for n in range(N):
    #    h[n] = h_t[:, n, :].reshape((T, H))
    h = h_t.transpose(1,0,2)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################

    N, T, H = dh.shape
    D = cache[0][0].shape[1]
    dh_loss = np.zeros((T, N, H))
    dx_t = np.zeros((T, N, D))
    dx = np.zeros((N, T, D))
    dprev_c = np.zeros((N, H))
    dprev_h = np.zeros((N, H))
    dWx = np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    db = np.zeros((4*H,))
    dnext_c = np.zeros((N,H))
    for t in range(T-1, -1, -1):
        dh_loss[t] = dh[:, t, :].reshape(N,H)
        dnext_h = dh_loss[t] + dprev_h
        dnext_c = dprev_c
        dx_temp, dprev_h, dprev_c, dWx_temp, dWh_temp, db_temp  = lstm_step_backward(dnext_h, dnext_c, cache[t])
        dx_t[t] += dx_temp
        dWx += dWx_temp
        dWh += dWh_temp
        db += db_temp
        if(t==0):
            print(dWh)

    dh0 = dprev_h
    dx = dx_t.transpose(1,0,2)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db

#________________________________________________________________________________________________________

def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

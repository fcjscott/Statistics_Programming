from builtins import range
import numpy as np


def fc_forward(x, w, b):
    out = x.reshape(x.shape[0], w.shape[0]).dot(w) + b
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    x, w, b = cache
    dw = x.reshape(x.shape[0], w.shape[0]).T.dot(dout)
    db = np.sum(dout, axis=0)
    dx = dout.dot(w.T).reshape(x.shape)
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = dout
    dx[cache < 0] = 0
    return dx


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    
    pool_height = pool_param.get('pool_height')
    pool_width = pool_param.get('pool_width')
    stride = pool_param.get('stride')
    N, C, H, W = x.shape

    out_H = np.int(((H - pool_height) / stride) + 1)
    out_W = np.int(((W - pool_width) / stride) + 1)
    out = np.zeros([N, C, out_H, out_W])

    for n in range(N): 
        for c in range(C): 
            for j in range(out_H):
                for i in range(out_W): 
                    out[n, c, j, i] = np.max(x[n,c, j*stride:j*stride+pool_height, i*stride:i*stride+pool_width])
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    
    x, pool_param = cache
    pool_height = pool_param.get('pool_height')
    pool_width = pool_param.get('pool_width')
    stride = pool_param.get('stride')
    
    N, C, H, W = x.shape
    _, _, dout_H, dout_W = dout.shape
    dx = np.zeros_like(x)

    for n in range(N): 
        for c in range(C):
            for j in range(dout_H):
                for i in range(dout_W):
                    max_idx = np.argmax(x[n,c, j*stride:j*stride+pool_height, i*stride:i*stride+pool_width])
                    max_position = np.unravel_index(max_idx, [pool_height,pool_width])
                    dx[n,c, j*stride:j*stride+pool_height, i*stride:i*stride+pool_width][max_position] = dout[n, c, j, i]

    return dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from pyfunt.layers.layers import log_softmax_loss

from pyfunt.layers.layer_utils import (affine_relu_forward,
                                       affine_relu_backward,
                                       affine_forward, affine_backward)

from pyfunt.layers.init import init_affine_wb


class FizzBuzzNet(object):

    def __init__(self, hidden_dims=[100], input_dim=10, num_classes=4, reg=0.0, weights=None, dtype=np.float32):
        '''
        Initialize the network
        '''
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.affine_layers = len(hidden_dims)

        self.softmax_l = self.affine_layers + 1
        self.affine_l = 1
        self.return_probs = False

        self.h_dims = hidden_dims

        if weights:
            for k, v in weights.iteritems():
                self.params[k] = v.astype(dtype)
            return

        self._init_affine_weights()

        self._init_scoring_layer(num_classes)

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def _init_affine_weights(self):
        '''
        Initialize affine weights.
        Called by self.__init__
        '''
        dims = [self.input_dim] + self.h_dims
        for i in xrange(self.affine_layers):
            idx = self.affine_l + i
            shape = dims[i], dims[i + 1]
            W, b = init_affine_wb(shape, 0.01)
            self.params['W%d' % idx] = W
            self.params['b%d' % idx] = b

    def _init_scoring_layer(self, num_classes):
        '''
        Initialize scoring layer weights.
        Called by self.__init__
        '''
        # Scoring layer
        in_ch = self.h_dims[-1]
        shape = in_ch, num_classes
        W, b = init_affine_wb(shape, 0.01)
        i = self.softmax_l
        self.params['W%d' % i] = W
        self.params['b%d' % i] = b

    def _extract(self, params, idx):
        '''
        Ectract Parameters from params
        '''
        w = params['W%d' % idx]
        b = params['b%d' % idx]
        return w, b

    def _put(self, cache, idx, h, cache_h):
        '''
        Put h and h_cache in cache
        '''
        cache['h%d' % idx] = h
        cache['cache_h%d' % idx] = cache_h
        return cache

    def _put_grads(self, cache, idx, dh, dw, db, ):
        '''
        Put grads in cache
        '''
        cache['dh%d' % (idx - 1)] = dh
        cache['dW%d' % idx] = dw
        cache['db%d' % idx] = db
        return cache

    def _forward_affines(self, cache):
        '''
        Execute affine layers's forward pass
        '''
        for i in xrange(self.affine_layers):
            idx = self.affine_l + i
            h = cache['h%d' % (idx - 1)]
            w, b = self._extract(self.params, idx)
            h, cache_h = affine_relu_forward(h, w, b)
            self._put(cache, idx, h, cache_h)

    def _forward_score_layer(self, cache):
        '''
        Execute softmax layer's forward pass
        '''
        idx = self.softmax_l
        w, b = self._extract(self.params, idx)
        h = cache['h%d' % (idx - 1)]
        h, cache_h = affine_forward(h, w, b)
        self._put(cache, idx, h, cache_h)

    def _backward_score_layer(self, dscores, cache):
        '''
        Execute softmax layer's backward pass
        '''
        idx = self.softmax_l
        dh = dscores
        h_cache = cache['cache_h%d' % idx]
        dh, dw, db = affine_backward(dh, h_cache)
        self._put_grads(cache, idx, dh, dw, db)

    def _backward_affines(self, cache):
        '''
        Execute affine layers' backward pass
        '''
        for i in range(self.affine_layers)[::-1]:
            idx = self.affine_l + i
            dh = cache['dh%d' % idx]
            h_cache = cache['cache_h%d' % idx]
            dh, dw, db = affine_relu_backward(
                dh, h_cache)
            self._put_grads(cache, idx, dh, dw, db)

    def loss_helper(self, args):
        '''
        Helper method used to call loss() within a pool of processes using \
        pool.map_async.
        '''
        return self.loss(*args)

    def loss(self, X, y=None, compute_dX=False):
        '''
        Evaluate loss and gradient for the three-layer convolutional network.
        '''
        X = X.astype(self.dtype)
        params = self.params
        scores = None

        cache = {}
        cache['h0'] = X

        self._forward_affines(cache)

        self._forward_score_layer(cache)

        scores = cache['h%d' % self.softmax_l]

        if y is None:
            if self.return_probs:
                probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                probs /= np.sum(probs, axis=1, keepdims=True)
                return probs

            return scores

        loss, grads = 0, {}
        data_loss, dscores = log_softmax_loss(scores, y)

        # Backward pass
        self._backward_score_layer(dscores, cache)

        self._backward_affines(cache)

        if compute_dX:
            return cache['dh0']

        # apply regularization to ALL parameters
        grads = {}
        reg_loss = .0
        for key, val in cache.iteritems():
            if key[:1] == 'd' and 'h' not in key:  # all params gradients
                reg_term = 0
                if self.reg:
                    reg_term = self.reg * params[key[1:]]
                    w = params[key[1:]]
                    reg_loss += 0.5 * self.reg * np.sum(w * w)
                grads[key[1:]] = val + reg_term

        loss = data_loss + reg_loss

        return loss, grads

import os, sys, re, itertools, time, cPickle
import librosa
import numpy as np
import scipy as sp
import scipy.io
import theano, theano.printing
import theano.tensor as T
import sklearn

tf = theano.config.floatX

def relu(x):
    if (re.search(r'theano', str(type(x)))):
        return T.maximum(0, x)
    else:
        return np.maximum(0, x)

class relu_layer:
    def __init__(self, n_in, n_out, rng):
        self.n_in = np.int32(n_in)
        self.n_out = np.int32(n_out)
        self.W = theano.shared(np.random.normal(0, 0.01, (n_out, n_in)).astype(tf), 'W')
        self.b = theano.shared(np.zeros(n_out, dtype=tf), 'b')
        self.rng = rng
        
    def prop(self, x):
        return relu(self.W.dot(x) + self.b[:, np.newaxis])

    def dropout_prop(self, x, delta):
        mask = 1.0 / (1.0 - delta) * \
               T.cast(self.rng.uniform(size=(self.n_out, x.shape[1]))
                      > delta, dtype=tf)
        return relu(self.W.dot(x) + self.b[:, np.newaxis]) * mask

    def dropconnect_prop(self, x, delta):
        mask = 1.0 / (1.0 - delta) * \
               T.cast(self.rng.uniform(size=(self.n_out, self.n_in))
                      > delta, dtype=tf)
        return relu((mask * self.W).dot(x) + self.b[:, np.newaxis])

    def squared_weights(self):
        return T.sum(self.W * self.W)

    def squared_biases(self):
        return T.sum(self.b * self.b)

class relu_series:
    def __init__(self, n_in, n_layers, h_size, rng):
        self.layers = []
        self.layers.append(relu_layer(n_in, h_size, rng))
        for i in xrange(1, n_layers):
            self.layers.append(relu_layer(h_size, h_size, rng))

    def prop(self, x):
        for layer in self.layers:
            x = layer.prop(x)
        return x

    def dropconnect_prop(self, x, delta):
        for layer in self.layers:
            x = layer.dropconnect_prop(x, delta)
        return x

    def dropout_prop(self, x, delta):
        for layer in self.layers:
            x = layer.dropout_prop(x, delta)
        return x

    def squared_weights(self):
        return np.sum(np.array([layer.squared_weights() for layer in self.layers]))

    def squared_biases(self):
        return np.sum(np.array([layer.squared_biases() for layer in self.layers]))

class relu_classifier:
    def __init__(self, n_in, n_layers, h_size, n_classes, rng):
        self.series = relu_series(n_in, n_layers, h_size, rng)
        self.alpha = theano.shared(np.zeros(n_classes, dtype=tf), 'alpha')
        self.beta = theano.shared(0.01*np.random.randn(n_classes, h_size).astype(tf), 'beta')
        self.x = theano.shared(np.zeros((n_in, 2), dtype=tf), 'x')
        self.y = theano.shared(np.zeros(10, dtype='int32'), 'y')
        self.eta = theano.shared(np.float32(1), 'eta')
        self.delta = theano.shared(np.float32(0), 'delta')

        logp = self.logp_t()
        grad = T.grad(logp, self.marshal_params())
        self.logp_f = theano.function([], logp)
        self.grad_f = theano.function([], grad)
        self.grad_update_f = theano.function([], [], updates=self.grad_updates())
        self.predict_f = theano.function([], self.predict_t())

    def marshal_params(self):
        result = []
        result.append(self.alpha)
        result.append(self.beta)
        for layer in self.series.layers:
            result.append(layer.W)
            result.append(layer.b)
        return result

    def predict_t(self):
        h = self.series.dropout_prop(self.x, self.delta)
#         h = self.series.dropconnect_prop(self.x, self.delta)
        result = self.alpha[:, np.newaxis] + self.beta.dot(h)
        result -= T.max(result, 0)
        return result - T.log(T.exp(result).sum(0))

    def logp_t(self):
        prediction = self.predict_t()
        return T.mean(prediction[self.y, T.arange(self.y.shape[0])])

    def grad_updates(self):
        result = []
        logp = self.logp_t()
        grad = T.grad(logp, self.marshal_params())
        result.append((self.alpha, self.alpha + self.eta * grad[0]))
        result.append((self.beta, self.beta + self.eta * grad[1]))
        i = 0
        for layer in self.series.layers:
            result.append((layer.W, layer.W + self.eta * grad[2+i*2]))
            result.append((layer.b, layer.b + self.eta * grad[3+i*2]))
            i += 1
        return result

    def set_xydelta(self, x, y, delta):
        self.x.set_value(x)
        self.y.set_value(y.astype('int32'))
        self.delta.set_value(delta)

    def logp(self, x, y, delta=0):
        self.set_xydelta(x, y, delta)
        return self.logp_f()

    def grad(self, x, y, delta=0):
        self.set_xydelta(x, y, delta)
        return self.grad_f()

    def predict(self, x, delta=0):
        self.x.set_value(x)
        self.delta.set_value(delta)
        return self.predict_f()

    def grad_step(self, x, y, eta, delta=0):
        self.set_xydelta(x, y, delta)
        self.eta.set_value(np.float32(eta))
        self.grad_update_f()

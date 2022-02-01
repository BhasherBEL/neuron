import numpy as np
from typing import Type
import time


##### COMPONENTS #####


class Component(object):
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.predict(inputs)

    def backward(self, dvalues):
        raise NotImplementedError()

    def predict(self, inputs):
        raise NotImplementedError()


##### LAYERS #####


class Layer(Component):
    def __init__(self):
        super().__init__()
        self.W = None
        self.B = None
        self.dW = None
        self.dB = None

    def backward(self, dvalues):
        super().backward(dvalues)

    def predict(self, inputs):
        super().predict(inputs)


class Layer_Dense(Layer):
    def __init__(self, n_inputs, n_neurons, mul=0.01):
        super().__init__()
        self.W = np.random.randn(n_inputs, n_neurons)*mul
        self.B = np.zeros((1, n_neurons))

    def backward(self, dvalues):
        self.dW = self.inputs.T @ dvalues
        self.dB = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = dvalues @ self.W.T

    def predict(self, inputs):
        return inputs @ self.W + self.B


##### ACTIVATIONS #####


class Activation(Component):
    def backward(self, dvalues):
        super().backward(dvalues)

    def predict(self, inputs):
        super().predict(inputs)


class Activation_Relu(Activation):
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predict(self, inputs):
        return np.maximum(0, inputs)


class Activation_LeakyRelu(Activation):
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] *= 0.05

    def predict(self, inputs):
        return np.maximum(inputs*0.05, inputs)


class Activation_Sigmoid(Activation):
    def backward(self, dvalues):
        self.dinputs = self.output*(1-self.output) * dvalues

    def predict(self, inputs):
        return 1/(1+np.exp(-inputs))


class Activation_Linear(Activation):
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predict(self, inputs):
        return inputs


class Activation_Tanh(Activation):
    def backward(self, dvalues):
        self.dinputs = (1-self.output**2) * dvalues

    def predict(self, inputs):
        return (np.exp(inputs)-np.exp(-inputs))/(np.exp(inputs)+np.exp(-inputs))


# class Activation_Softmax(Activation):
#     def forward(self, inputs):
#         super().forward(inputs)
#         exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
#         self.output = exp/np.sum(exp, axis=1, keepdims=True)


##### LOSS #####


class Loss(object):
    def __init__(self):
        self.dinputs = None
        self.losses = None

    def score(self, y_pred, y):
        return np.mean(self.forward(y_pred, y))

    def forward(self, y_pred, y):
        raise NotImplementedError()

    def backward(self, dvalues, y):
        raise NotImplementedError()


# class Loss_CategoricalCrossEntropy(Loss):
#     def forward(self, y_pred, y):
#         return -np.log(np.clip(y_pred, 1e-10, 1-1e-10)[range(len(y_pred)), y])
#
#     def backward(self, dvalues, y):
#         y = np.eye(len(dvalues[0]))[y]
#
#         self.dinputs = -y/dvalues/len(dvalues)


# class Activation_Softmax_Loss_CategoricalCrossEntropy:
#
#     def __init__(self):
#         self.activation = Activation_Softmax()
#         self.loss = Loss_CategoricalCrossEntropy()
#         self.inputs = None
#         self.output = None
#         self.dinputs = None
#
#     def forward(self, inputs, y):
#         self.activation.forward(inputs)
#         self.output = self.activation.output
#         return self.loss.calculate(self.output, y)
#
#     def backward(self, dvalues, y):
#         self.dinputs = dvalues.copy()
#         self.dinputs[range(len(dvalues)), y] -= 1
#         self.dinputs /= len(dvalues)


class MSE(Loss):
    def forward(self, y_pred, y):
        return np.sum((y_pred-y)**2, axis=1)

    def backward(self, y_pred, y):
        self.dinputs = 2*(y_pred-y)


##### OPTIMIZERS #####


class Optimizer(object):
    def __init__(self, learning_rate=1e-6):
        self.learning_rate = self.real_learning_rate = learning_rate

    def pre_update_params(self):
        pass

    def update_params(self, layer: Layer):
        raise NotImplementedError()

    def post_update_params(self):
        pass


class Optimizer_SGD(Optimizer):
    def update_params(self, layer: Layer):
        layer.W -= self.real_learning_rate * layer.dW
        layer.B -= self.real_learning_rate * layer.dB


class Optimizer_SGDM(Optimizer):
    def __init__(self, learning_rate=1e-6, momentum=.5, decay=1e-7):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.decay = decay
        self.it = 0

    def pre_update_params(self):
        if self.decay:
            self.real_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.it))

    def update_params(self, layer: Layer):
        if not hasattr(layer, 'Wm'):
            layer.Wm = np.zeros_like(layer.W)
            layer.Bm = np.zeros_like(layer.B)

        layer.Wm = self.momentum * layer.Wm - self.real_learning_rate * layer.dW
        layer.Bm = self.momentum * layer.Bm - self.real_learning_rate * layer.dB

        layer.W += layer.Wm
        layer.B += layer.Bm

    def post_update_params(self):
        self.it += 1


class Optimizer_Adam(Optimizer):
    def __init__(self, learning_rate=1e-5, decay=1e-4, epsilon=1e-7, beta1=.9, beta2=.999):
        super().__init__(learning_rate=learning_rate)
        self.real_learning_rate = learning_rate
        self.decay = decay
        self.it = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def pre_update_params(self):
        if self.decay:
            self.real_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.it))

    def update_params(self, layer: Layer):
        if not hasattr(layer, 'Wc'):
            layer.Wm = np.zeros_like(layer.W)
            layer.Wc = np.zeros_like(layer.W)
            layer.Bm = np.zeros_like(layer.B)
            layer.Bc = np.zeros_like(layer.B)

        layer.Wm = self.beta1 * layer.Wm + (1-self.beta1) * layer.dW
        layer.Bm = self.beta1 * layer.Bm + (1-self.beta1) * layer.dB

        layer.Wc = self.beta2 * layer.Wc + (1-self.beta2) * layer.dW**2
        layer.Bc = self.beta2 * layer.Bc + (1-self.beta2) * layer.dB**2

        wmc = layer.Wm/(1-self.beta1**(self.it+1))
        bmc = layer.Bm/(1-self.beta1**(self.it+1))

        layer.Wc = self.beta2 * layer.Wc + (1-self.beta2) * layer.dW**2
        layer.Bc = self.beta2 * layer.Bc + (1-self.beta2) * layer.dB**2

        wcc = layer.Wc/(1-self.beta2**(self.it+1))
        bcc = layer.Bc/(1-self.beta2**(self.it+1))

        layer.uW = - self.real_learning_rate * wmc
        layer.uB = - self.real_learning_rate * bmc

        layer.W += layer.uW  # / (wcc**2 + self.epsilon)
        layer.B += layer.uB  # / (bcc**2 + self.epsilon)

    def post_update_params(self):
        self.it += 1


##### NORMALIZERS #####

class Normalizer(Component):
    def forward(self, inputs):
        super().forward(inputs)

    def backward(self, outputs):
        super().backward(outputs)

    def predict(self, inputs):
        return self.forward(inputs)


class Normalizer_MinMax(Normalizer):
    def __init__(self, inputs):
        super().__init__()
        self.subtract = np.min(inputs, axis=0)
        self.divider = np.max(inputs, axis=0) - np.min(inputs, axis=0)

    def forward(self, inputs):
        return (inputs-self.subtract)/self.divider

    def backward(self, outputs):
        return outputs*self.divider+self.subtract


##### NETWORKS #####


class Network_Dense(Component):
    def __init__(self, sizes: list, activation: Type[Activation] = Activation_Tanh,
                 final_activation: Type[Activation] = Activation_Linear,
                 optimizer: Type[Optimizer] = Optimizer_SGD, learning_rate=None,
                 loss: Type[Loss] = MSE):
        super().__init__()
        self.layers: list[Layer] = [Layer_Dense(s1, s2) for s1, s2 in zip(sizes[:-1], sizes[1:])]
        self.activations: list[Activation] = [activation() for _ in sizes[1:-1]]
        self.activations.append(final_activation())
        self.optimizer = optimizer(learning_rate) if learning_rate else optimizer()
        self.loss = loss()
        self.losses = []

    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
        for layer, activation in zip(self.layers, self.activations):
            layer.forward(self.output)
            activation.forward(layer.output)
            self.output = activation.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.optimizer.pre_update_params()
        for layer, activation in zip(reversed(self.layers), reversed(self.activations)):
            activation.backward(self.dinputs)
            layer.backward(activation.dinputs)
            self.optimizer.update_params(layer)
            self.dinputs = layer.dinputs
        self.optimizer.post_update_params()

    def predict(self, inputs):
        output = inputs
        for layer, activation in zip(self.layers, self.activations):
            output = activation.predict(layer.predict(output))
        return output

    def train_once(self, inputs, outputs):
        self.forward(inputs)
        self.loss.backward(self.output, outputs)
        self.losses.append(self.loss.score(self.output, outputs))
        self.backward(self.loss.dinputs)

    def train(self, inputs, outputs, epochs=10000, learning_rate=None):
        if learning_rate:
            self.optimizer.learning_rate = learning_rate
        st = time.time()
        epoch_per_second = 10
        done = 0
        initial_loss = self.loss.score(self.predict(inputs), outputs)
        print(f'epoch:\t0, loss: {initial_loss:.5f} (0.00%)           ', end='')
        while done < epochs:
            for i in range(epoch_per_second):
                self.train_once(inputs, outputs)
            done += epoch_per_second
            duration = time.time()-st
            epoch_per_second = min(max(int(done/duration), 1), epochs-done)
            print(f'\repoch:\t{done}, loss: {self.losses[-1]:.5f} ({self.losses[-1]/initial_loss-1:.2%}), time: {duration:.0f}/{duration*epochs/done:.0f}s           ', end='')


#### SAMPLE ####


def sample_area(center, variance, amount=50):
    return center + np.random.randn(amount, len(center)) * variance

"""
Class for different layers
"""
import numpy as np
import scipy as sp
from abc import ABC, abstractmethod

class Layer(ABC):
    """
    """
    @abstractmethod
    def forward(self,
        x:  np.array,
    ) -> np.array:
        pass

    @abstractmethod
    def grad(self,
        grads:  np.array,
    ) -> None:
        pass

class Linear(Layer):
    """
    """
    def __init__(self,
        inputs:     int,
        outputs:    int,
        parent,
        bias:       bool=False,
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.weights = np.zeros(shape=(outputs,inputs))
        if bias:
            self.bias = np.zeros(1)
        else:
            self.bias = None
        self.vals = None

    def forward(self,
        x:  np.array,
    ):
        assert x.shape[0] == self.inputs
        self.vals = x
        if self.bias:
            return self.weights * x + self.bias
        return self.weights * x
    
    def grad(self,
        grads:  np.array,
    ):
        assert grads.shape[1] == self.weights.shape[0]
        assert grads.shape[0] == self.vals.shape[1]
        loss = grads * self.weights
        self.weights -= loss.transpose() * self.vals
        if self.parent == None:
            return
        else:
            return self.parent.grad(loss)

class Tanh(Layer):
    """
    """
    def __init__(self,
        inputs: int,
        parent,
    ) -> None:
        self.inputs
        self.parent = parent
        self.vals = None

    def forward(self,
        x:  np.array,
    ) -> np.array:
        assert x.shape[0] == self.inputs
        self.vals = x
        return np.tanh(x)
    
    def grad(self,
        grads:  np.array,
    ) -> None:
        loss = (1. - self.vals**2) * grads
        return self.parent.grad(loss)
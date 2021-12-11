"""
Class for a generic MLP
"""
import numpy as np
import scipy as sp
from abc import ABC, abstractmethod
from layer import *

class MLP(ABC):

    @abstractmethod
    def forward(self,
        x:  np.array,
    ) -> None:
        """The forward pass through the network"""
        pass
    
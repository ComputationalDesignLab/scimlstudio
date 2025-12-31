from abc import abstractmethod
from torch import nn

class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

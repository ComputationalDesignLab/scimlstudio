from abc import ABC, abstractmethod
from .base_model import BaseModel

class BaseDimensionalityReduction(BaseModel):

    @abstractmethod
    def encoding(self):
        pass

    @abstractmethod
    def decoding(self):
        pass


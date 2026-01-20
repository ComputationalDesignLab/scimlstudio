from abc import abstractmethod
from .base_model import BaseModel
import torch

class GPBaseModel(BaseModel, torch.nn.Module):

    @abstractmethod
    def posterior(self):
        pass

    @abstractmethod
    def transform_inputs(self):
        pass

    @abstractmethod
    def transform_outputs(self):
        pass

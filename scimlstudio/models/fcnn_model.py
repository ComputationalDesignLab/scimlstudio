import torch
from ..base_models import BaseModel
from ..utils import Standardize, Normalize

class FeedForwardNeuralNetwork(BaseModel):

    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        network: torch.nn.Sequential,
        input_transform: Standardize | Normalize | None = None,
        output_transform: Standardize | Normalize | None = None,
    ):
        """
            Class definition for training and predicting using 
            feed-forward neural network for supervised problems

            Parameters
            ----------
            x_train: torch.Tensor
                Input training data for the network in a 2D tensor

            y_train: torch.Tensor
                Output training data for the network in a 2D tensor

            network: torch.nn.Sequential
                Sequential object defining the network

            input_transform: Normalize or Standardize or None
                Data scaling class for the inputs of the network

            output_transform: Normalize or Standardize or None
                Data scaling class for the outputs of the network
        """

        # Some checks
        assert isinstance(x_train, torch.Tensor) and x_train.ndim == 2, "xtrain must be a 2D tensor array"
        assert isinstance(y_train, torch.Tensor) and y_train.ndim == 2, "ytrain must be a 2D tensor array"
        assert x_train.shape[0] == y_train.shape[0], "number of samples in input and output training data must be the same"
        assert x_train.device == y_train.device, "input and output training data must be on the same device"
        assert isinstance(network, torch.nn.Sequential), "network should be an instance of sequential class from torch.nn module"
        for param in network.parameters():
            assert param.device == x_train.device, "network parameters should be on the same device as the training data"

        if input_transform is not None:
            assert isinstance(input_transform, Normalize) or isinstance(input_transform, Standardize), "input_transform should be an instance of Normalize or Standardize class"

        if output_transform is not None:
            assert isinstance(output_transform, Normalize) or isinstance(output_transform, Standardize), "output_transform should be an instance of Normalize or Standardize class"

        try:
            network.eval()
            with torch.no_grad():
                network(x_train[0])
        except Exception as e:
            raise RuntimeError(f"Network architecture is not compatible with the provided data: {e}")

        super().__init__()

        self.x_train = x_train
        self.y_train = y_train
        self.network = network
        self.input_transform = input_transform
        self.output_transform = output_transform

    def fit(self):

        pass

    def predict(self):

        pass

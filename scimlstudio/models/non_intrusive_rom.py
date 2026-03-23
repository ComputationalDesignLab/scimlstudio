import torch
from ..base_models import BaseModel
from .feed_forward_autoencoder import FeedForwardAutoencoder
from .proper_orthogonal_decomposition import POD
from .feed_forward_nn_model import FeedForwardNeuralNetwork
from ..utils import Standardize, Normalize

class NonIntrusiveReducedOrderModel(BaseModel):

    def __init__(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            dim_red_model: FeedForwardAutoencoder | POD,
            latent_space_model: FeedForwardNeuralNetwork,
            input_transform: Standardize | Normalize | None = None,
            output_transform: Standardize | Normalize | None = None      
    ):
        """
            Definition of the class for a non-intrusive reduced order models (NIROMs)

            Parameters
            ----------
            x_train: torch.Tensor
                Input data for the NIROM
            y_train: torch.Tensor
                Output data for the NIROM
            dim_red_model: FeedForwardAutoencoder or POD
                An instance of the FeedForwardAutoencoder or POD class
                to be used as the dimensionality reduction model
            latent_space_model: FeedForwardNeuralNetwork
                An instance of the FeedForwardNeuralNetwork class to 
                be used as the latent space model of the 
            input_transform: Standardize or Normalize or None
                Transformation to apply to the input data 
            output_transform: Standardize or Normalize or None
                Transformation to apply to the output data 
        """
        super().__init__()

        # Setting the data
        assert isinstance(x_train, torch.Tensor) and x_train.ndim == 2, "input data must be provided as a 2D tensor array"
        assert isinstance(y_train, torch.Tensor) and y_train.ndim == 2, "output data must be provided as a 2D tensor array"
        self.x_train = x_train
        self.y_train = y_train

        # Setting the snapshot transform
        if input_transform is not None:
            assert isinstance(input_transform, Normalize) or isinstance(input_transform, Standardize), "input transform must be an instance of Normalize or Standardize class"
        if output_transform is not None:
            assert isinstance(output_transform, Normalize) or isinstance(output_transform, Standardize), "input transform must be an instance of Normalize or Standardize class"
        self.input_transform = input_transform

        # Checking the dimensionality reduction model
        assert isinstance(dim_red_model, FeedForwardAutoencoder) or isinstance(dim_red_model, POD), "the dimensionality reduction model must be an instance of the FeedForwardAutoencoder or POD class"
        self.dim_red_model = dim_red_model

        # Checking the latent space model
        assert isinstance(latent_space_model, FeedForwardNeuralNetwork), "the latent space model must be an instance of the FeedForwardNeuralNetwork class"
        self.latent_space_model = latent_space_model

        


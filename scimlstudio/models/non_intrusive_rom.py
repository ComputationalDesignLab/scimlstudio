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
        """
        super().__init__()

        # Setting the data
        assert isinstance(x_train, torch.Tensor) and x_train.ndim == 2, "input data must be provided as a 2D tensor array"
        assert isinstance(y_train, torch.Tensor) and y_train.ndim == 2, "output data must be provided as a 2D tensor array"
        self.x_train = x_train
        self.y_train = y_train

        # Checking the dimensionality reduction model
        assert isinstance(dim_red_model, FeedForwardAutoencoder) or isinstance(dim_red_model, POD), "the dimensionality reduction model must be an instance of the FeedForwardAutoencoder or POD class"
        self.dim_red_model = dim_red_model

        # Checking the latent space model
        assert isinstance(latent_space_model, FeedForwardNeuralNetwork), "the latent space model must be an instance of the FeedForwardNeuralNetwork class"
        self.latent_space_model = latent_space_model

    def fit(
        self,
        latent_space_optimizer: torch.optim.Optimizer,
        loss_func: torch.nn.modules.loss._Loss,
        dim_red_optimizer: torch.optim.Optimizer | None,
        dim_red_epochs: int | None = None,
        batch_size: int = 1,
        latent_space_epochs: int = 100,
        convert_to_eval_mode: bool = True      
    ):
        """
            Method to fit the NIROM to the training data
            
            `NOTE`: This method supports mini-batch training

            Parameters
            ----------
            latent_space_optimizer: torch.optim.Optimizer
                Optimizer object from torch.optim module to optimize the parameters of the latent space neural network

            loss_func: torch.nn.modules.loss._Loss
                Loss function object from torch.nn.Module.loss module to compute the loss during training
                This is used for both the latent space model and dimensionality reduction model, if the 
                dimensionality reduction model is an autoencoder. 

            batch_size: int
                Batch size to use during training, default = 1

            dim_red_optimizer: torch.optim.Optimizer
                Optimizer object from torch.optim module to optimize the autoencoder parameters, if applicable
                If POD is being used, this does not need to be provided and will be set to None as a default.

            dim_red_epochs: int
                Number of epochs to train the dimensionality reduction method if an autoencoder is used, default = 100
                If POD is being used, this does not need to be provided and will be set to None as a default.

            latent_space_epochs: int
                Number of epochs to train the latent space model, default = 100

            convert_to_eval_mode: bool
                Flag to set the neural networks within the NIROM to eval mode after training is done, default = True
        """

        assert isinstance(latent_space_optimizer, torch.optim.Optimizer), "`optimizer` should be an instance of PyTorch optimizer class"
        assert isinstance(loss_func, torch.nn.modules.loss._Loss), "`loss_func` should be an instance of a PyTorch loss function class"
        assert isinstance(batch_size, int) and batch_size > 0, "`batch_size` should be a positive integer"
        assert isinstance(latent_space_epochs, int) and latent_space_epochs > 0, "`latent_space_epochs` should be a positive integer"
        assert isinstance(convert_to_eval_mode, bool), "`convert_to_eval_mode` should be a boolean value"

        if isinstance(self.dim_red_model, POD):
            self.dim_red_model.fit()
            latent_vars = self.dim_red_model.encoding(self.y_train)
        
        if isinstance(self.dim_red_model, FeedForwardAutoencoder):
            assert isinstance(dim_red_optimizer, torch.optim.Optimizer), "`dim_red_optimizer` should be an instance of PyTorch optimizer class since autoencoder is being used"
            assert isinstance(dim_red_epochs, int) and dim_red_epochs > 0, "`dim_red_epochs` should be a positive integer if autoencoder is being used"
            self.dim_red_model.fit(optimizer=dim_red_optimizer, loss_func=loss_func, batch_size=batch_size, epochs=dim_red_epochs)
            latent_vars = self.dim_red_model.encoding(self.y_train)

            







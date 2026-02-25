import torch
from ..base_models import BaseDimensionalityReduction
from ..utils import Standardize, Normalize

class FeedForwardAutoencoder(BaseDimensionalityReduction):

    def __init__(
            self,
            x_train: torch.Tensor,
            encoder: torch.nn.Sequential,
            decoder: torch.nn.Sequential,
            data_transform: Standardize | Normalize | None = None
    ):
        """
            Definition of the class for feedforward autoencoder neural networks

            Parameters
            ----------
            x_train: torch.Tensor
                High-dimensional input data for the autoencoder
            encoder: torch.nn.Sequential
                Sequential object for defining the encoder network
            decoder: torch.nn.Sequential
                Sequential object for defining the decoder network
            data_transform: Standardize or Normalize or None
                Transformation to apply to the high-dimensional input data 

        """
        super().__init__()

        # Setting the input data
        assert isinstance(x_train, torch.Tensor) and x_train.dim == 2, "high dimensional input data must be provided as a 2D tensor array"
        self.s_train = x_train

        # Setting the snapshot transform
        if data_transform is not None:
            assert isinstance(data_transform, Normalize) or isinstance(data_transform, Standardize), "data transform must be an instance of Normalize or Standardize class"
        self.data_transform = data_transform

        # Setting the encoder and decoder
        assert isinstance(encoder, torch.nn.Sequential), "encoder network should be an instance of sequential class from torch.nn module"
        for param in encoder.parameters():
            assert param.device == x_train.device, "encoder parameters should be on the same device as the training data"
        
        assert isinstance(decoder, torch.nn.Sequential), "decoder network should be an instance of sequential class from torch.nn module"
        for param in decoder.parameters():
            assert param.device == x_train.device, "decoder parameters should be on the same device as the training data"

        try:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                encoded = encoder(x_train[0])
                decoded = decoder(encoded)
        except Exception as e:
            raise RuntimeError(f"Network architecture is not correct and/or not compatible with the provided data: {e}")
        
        # Setting the two networks in training mode
        encoder.train()
        decoder.train()

        self.decoder = decoder
        self.encoder = encoder

    def fit(
            self,
            optimizer: torch.optim.Optimizer,
            epochs: int = 100,
            batch_size: int = 1, 
            loss_func = torch.nn.modules.loss._Loss,
            convert_to_eval_mode = True,
    ):
        """
            Class method to fit the autoencoder neural network to the data
            Similar to fit method for the feed forward neural network model

            `NOTE`: This method supports mini-batch training

            Parameters
            ----------
            optimizer: torch.optim.Optimizer
                Optimizer object from torch.optim module to optimize the network parameters

            loss_func: torch.nn.modules.loss._Loss
                Loss function object from torch.nn.Module.loss module to compute the loss during training

            batch_size: int
                Batch size to use during training, default = 1

            epochs: int
                Number of epochs to train the network, default = 100

            convert_to_eval_mode: bool
                Flag to set the network to eval mode after training is done, default = True
        """

        assert isinstance(optimizer, torch.optim.Optimizer), "`optimizer` should be an instance of PyTorch optimizer class"
        assert isinstance(loss_func, torch.nn.modules.loss._Loss), "`loss_func` should be an instance of a PyTorch loss function class"
        assert isinstance(batch_size, int) and batch_size > 0, "`batch_size` should be a positive integer"
        assert isinstance(epochs, int) and epochs > 0, "`epochs` should be a positive integer"
        assert isinstance(convert_to_eval_mode, bool), "`convert_to_eval_mode` should be a boolean value"
    
        # applying the transform to the high dimensional data
        if self.data_transform is not None:
            x_train = self.data_transform.transform(self.x_train)
        else:
            x_train = self.x_train

        # dataset and dataloader
        dataset = torch.utils.data.TensorDataset(x_train, x_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # training loop
        for epoch in range(epochs):

            # loop over all batches
            for x_batch, y_reconstruction in dataloader:

                optimizer.zero_grad() # zero the grads

                latent = self.encoder(x_batch) # encoding the data

                reconstruction = self.decoder(latent) # decoding the data 

                loss = loss_func(reconstruction, y_reconstruction) # compute the loss

                loss.backward() # backward pass

                optimizer.step() # update the parameters

        if convert_to_eval_mode:
            self.encoder.eval()
            self.decoder.eval()

    def encoding(self, x: torch.Tensor):
        """
            Class method to encode data into the latent space using the encoder network

            Parameters
            ----------
            x: torch.Tensor
                2D Tensor array containing the data to be encoded

            Returns
            -------
            latent: torch.Tensor
                2D Tensor array containing the latent space coordinates for the data provided
        """
        assert isinstance(x, torch.Tensor) and x.ndim==2, "x must be 2D Tensor array"
        assert x.device == self.x_train.device, "the given data must be on the same device as the training data"
        # Transforming the input
        if self.data_transform is not None:
            x = self.data_transform.transform(x)
        # Calculating the latent space coordinates
        latent = self.encoder(x)
        return latent
    
    def decoding(self, latent: torch.Tensor):
        """
            Class method to decode latent space to recover the original 
            high-dimensional data

            Parameters
            ----------
            latent: torch.Tensor
                2D Tensor array containing the latent space coordinates

            Returns
            -------
            y : torch.Tensor
                2D Tensor array containing reconstructions of the high dimensional data
        """
        assert isinstance(latent, torch.Tensor) and latent.ndim==2, "corodinates must be 2D Tensor array"
        assert latent.device == self.x_train.device, "the given data must be on the same device as the training snapshots"
        # Calculating the reconstruction
        y = self.decoder(latent)
        # Transforming the reconstruction
        if self.data_transform is not None:
            y = self.data_transform.inverse_transform(y)
        return y
    
    def predict(self, x: torch.Tensor):
        """
            Class method for prediction using the autoencoder model

            Parameters
            ----------
            x: torch.Tensor
                2D Tensor array containing the high-dimensional data

            Returns
            -------
            reconstruction: torch.Tensor
                2D Tensor array containing the reconstructions of the data
        """
        assert isinstance(x, torch.Tensor) and x.ndim==2, "x must be 2D Tensor array"
        assert x.device == self.x_train.device, "the given data must be on the same device as the training snapshots"

        with torch.no_grad():
            if self.data_transform is not None:
                x = self.data_transform.transform(x)
            encoding = self.encoder(x)
            reconstruction = self.decoder(encoding)
            if self.data_transform is not None:
                reconstruction = self.data_transform.inverse_transform(reconstruction)
            return reconstruction
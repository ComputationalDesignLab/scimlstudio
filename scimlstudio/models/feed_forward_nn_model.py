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
            a feed-forward neural network for supervised problems

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
            raise RuntimeError(f"Network architecture is not correct and/or not compatible with the provided data: {e}")

        super().__init__()

        network.train() # set network in train mode

        self.x_train = x_train
        self.y_train = y_train
        self.network = network
        self.input_transform = input_transform
        self.output_transform = output_transform

    @property
    def parameters(self):
        return self.network.parameters() # network parameters

    def fit(
        self,
        optimizer: torch.optim.Optimizer,
        loss_func: torch.nn.modules.loss._Loss,
        batch_size: int = 1,
        epochs: int = 100,
        convert_to_eval_mode: bool = True
    ):
        """
            Method to fit the network to the training data

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
        
        self.network.train() # set network in train mode

        # transform the training data
        if self.input_transform is not None:
            x_train = self.input_transform.transform(self.x_train)
        else:
            x_train = self.x_train

        if self.output_transform is not None:            
            y_train = self.output_transform.transform(self.y_train)
        else:
            y_train = self.y_train

        # dataset and dataloader
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # training loop
        for epoch in range(epochs):

            # loop over all batches
            for x_batch, y_batch in dataloader:

                optimizer.zero_grad() # zero the grads

                y_pred = self.network(x_batch) # forward pass

                loss = loss_func(y_pred, y_batch) # compute the loss

                loss.backward() # backward pass

                optimizer.step() # update the parameters

        if convert_to_eval_mode:
            self.network.eval()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
            Method to predict the output for the given input data

            `NOTE`: predictions are made in no grad context

            Parameters
            ----------
            x: torch.Tensor
                a torch tensor representing the input data used for prediction

            Returns
            -------
            y_pred: torch.Tensor
                a torch tensor representing the predicted output for the given input data
        """

        assert isinstance(x, torch.Tensor), "`x` should be a torch tensor"
        assert x.device == self.x_train.device, "input data should be on the same device as the training data"

        x_ndim = x.ndim # number of dimensions in the given input data

        # check input shape and add batch dim if necessary
        if x_ndim == self.x_train.ndim:
            assert x.shape[1:] == self.x_train.shape[1:], "input data should have the same feature size as the training data"
        elif x_ndim == self.x_train.ndim - 1:
            assert x.shape == self.x_train.shape[1:], "input data should have the same feature size as the training data"
            x = x.unsqueeze(0) # add batch dim as 1
        else:
            raise ValueError("input data should be of similar shape as the training data")
        
        # check if network is in train mode
        if self.network.training:
            raise RuntimeError("Network is in train mode, please use the `fit` method to train the network first and then call the `predict` method")
        
        # transform the input data
        if self.input_transform is not None:
            x = self.input_transform.transform(x)

        # predict in no grad context
        with torch.no_grad():
            y_pred = self.network(x)

        # inverse transform the predicted output
        if self.output_transform is not None:
            y_pred = self.output_transform.inverse_transform(y_pred)

        # remove batch dim, if it was added
        if x_ndim == self.x_train.ndim - 1:
            y_pred = y_pred.squeeze(0)

        return y_pred

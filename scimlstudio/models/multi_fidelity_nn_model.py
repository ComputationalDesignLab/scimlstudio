import torch
from ..base_models import BaseModel
from ..utils import Standardize, Normalize
from itertools import chain

class MultifidelityNeuralNetwork(BaseModel):

    def __init__(
        self,
        x_train_lf: torch.Tensor,
        y_train_lf: torch.Tensor,
        x_train_hf: torch.Tensor,
        y_train_hf: torch.Tensor,
        network_lf: torch.nn.Sequential,
        network_linear_corr: torch.nn.Sequential,
        network_nonlinear_corr: torch.nn.Sequential,
        input_transform: Standardize | Normalize | None = None,
        output_transform_lf: Standardize | Normalize | None = None,
        output_transform_hf: Standardize | Normalize | None = None,
    ):
        """
            Class providing an implementation of multi-fidelity neural networks. This class 
            uses composite neural networks formulation proposed by Meng and Karniadakis:
            https://www.sciencedirect.com/science/article/pii/S0021999119307260

            `NOTE`: Current implementation only works for bi-fidelity problems

            Parameters
            ----------
            x_train_lf: torch.Tensor
                Input low fidelity training data for the network in a 2D tensor

            y_train_lf: torch.Tensor
                Output low fidelity training data for the network in a 2D tensor

            x_train_hf: torch.Tensor
                Input high fidelity training data for the network in a 2D tensor

            y_train_hf: torch.Tensor
                Output high fidelity training data for the network in a 2D tensor

            network_lf: torch.nn.Sequential
                Sequential object defining the network for low-fidelity data

            network_linear_corr: torch.nn.Sequential
                Sequential object defining the network for linear correlation

            network_nonlinear_corr: torch.nn.Sequential
                Sequential object defining the network for nonlinear correlation

            input_transform: Normalize or Standardize or None
                Instance of data scaling class for the input

            output_transform_lf: Normalize or Standardize or None
                Instance of data scaling class for the low fidelity output

            output_transform_hf: Normalize or Standardize or None
                Instance of data scaling class for the high fidelity output
        """

        # Some checks
        for x_train, y_train, fidelity in zip([x_train_lf, x_train_hf], [y_train_lf, y_train_hf], ["low fidelity", "high fidelity"]):
            assert isinstance(x_train, torch.Tensor) and x_train.ndim == 2, f"xtrain {fidelity} must be a 2D tensor array"
            assert isinstance(y_train, torch.Tensor) and y_train.ndim == 2, f"ytrain {fidelity} must be a 2D tensor array"
            assert x_train.shape[0] == y_train.shape[0], f"number of samples in input and output {fidelity} training data must be the same"
            assert x_train.device == y_train.device, f"input and output {fidelity} training data must be on the same device"
        assert x_train_lf.device == x_train_hf.device, "high and low fidelity data should be on same device"

        for network, net_type in zip([network_lf, network_linear_corr, network_nonlinear_corr], ["low fidelity", "linear correlation", "nonlinear correlation"]):
            assert isinstance(network, torch.nn.Sequential), f"{net_type} network should be an instance of sequential class from torch.nn module"

            for param in network.parameters():
                assert param.device == x_train_lf.device, "network parameters should be on the same device as the training data"

            network.train() # set network in train mode

        if input_transform is not None:
            assert isinstance(input_transform, Normalize) or isinstance(input_transform, Standardize), f"{net_type} input_transform should be an instance of Normalize or Standardize class"

        for output_transform, net_type in zip([output_transform_lf, output_transform_hf], ["low fidelity", "high fidelity"]):
            if output_transform is not None:
                assert isinstance(output_transform, Normalize) or isinstance(output_transform, Standardize), f"{net_type} output_transform should be an instance of Normalize or Standardize class"

        super().__init__()

        self.input_transform = input_transform

        # low fidelity
        self.x_train_lf = x_train_lf
        self.y_train_lf = y_train_lf
        self.output_transform_lf = output_transform_lf

        # high fidelity
        self.x_train_hf = x_train_hf
        self.y_train_hf = y_train_hf
        self.output_transform_hf = output_transform_hf

        # networks
        self.network_lf = network_lf
        self.network_linear_corr = network_linear_corr
        self.network_nonlinear_corr = network_nonlinear_corr

    # network parameters
    @property
    def parameters(self):
        return chain(self.network_lf.parameters(), self.network_linear_corr.parameters(), self.network_nonlinear_corr.parameters())

    def fit(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int = 100,
        reg_const_lf: float = 1e-4,
        reg_const_linear: float = 0.0,
        reg_const_nonlinear: float = 1e-3,
        convert_to_eval_mode: bool = True
    ):
        """
            Method to fit the network to the training data

            `NOTE`: This method supports mini-batch training

            Parameters
            ----------
            optimizer: torch.optim.Optimizer
                Optimizer object from torch.optim module to optimize the network parameters

            epochs: int
                Number of epochs to train the network, default = 100

            reg_const_lf: float
                regularization constant for low fidelity network, default = 1e-4

            reg_const_linear: float
                regularization constant for linear correlation network, default = 0.0

            reg_const_nonlinear: float
                regularization constant for nonlinear correlation network, default = 1e-3

            convert_to_eval_mode: bool
                Flag to set the network to eval mode after training is done, default = True
        """

        assert isinstance(optimizer, torch.optim.Optimizer), "`optimizer` should be an instance of PyTorch optimizer class"
        assert isinstance(epochs, int) and epochs > 0, "`epochs` should be a positive integer"
        assert isinstance(convert_to_eval_mode, bool), "`convert_to_eval_mode` should be a boolean value"
        
        networks = [self.network_lf, self.network_linear_corr, self.network_nonlinear_corr]

        for network in networks:
            network.train() # set network in train mode

        # loss function
        loss_func = torch.nn.MSELoss()

        if self.input_transform is not None:
            x_train_lf = self.input_transform.transform(self.x_train_lf)
            x_train_hf = self.input_transform.transform(self.x_train_hf)
        else:
            x_train_lf = self.x_train_lf
            x_train_hf = self.x_train_hf

        if self.output_transform_lf is not None:
            y_train_lf = self.output_transform_lf.transform(self.y_train_lf)
        else:
            y_train_lf = self.y_train_lf

        if self.output_transform_hf is not None:
            y_train_hf = self.output_transform_hf.transform(self.y_train_hf)
        else:
            y_train_hf = self.y_train_hf

        # training loop
        for epoch in range(epochs):

            optimizer.zero_grad() # zero the grads

            y_train_lf_pred = self.network_lf(x_train_lf) # forward pass - low fidelity network

            y_train_hf_lf_pred = self.network_lf(x_train_hf) # forward pass - low fidelity network

            x_train_hf_corr = torch.hstack((x_train_hf, y_train_hf_lf_pred)) # combined input for correlation networks

            y_train_hf_linear_corr_pred = self.network_linear_corr(x_train_hf_corr) # compute linear correlation

            y_train_hf_nonlinear_corr_pred = self.network_nonlinear_corr(x_train_hf_corr) # compute nonlinear correlation

            y_train_hf_pred = y_train_hf_linear_corr_pred + y_train_hf_nonlinear_corr_pred

            loss_lf = loss_func(y_train_lf_pred, y_train_lf) # compute low fidelity loss

            loss_hf = loss_func(y_train_hf_pred, y_train_hf) # compute high fidelity loss

            # regularization
            reg_lf = reg_const_lf * sum([(param**2).sum() for param in self.network_lf.parameters()])
            reg_linear_corr = reg_const_linear * sum([(param**2).sum() for param in self.network_linear_corr.parameters()])
            reg_nonlinear_corr = reg_const_nonlinear * sum([(param**2).sum() for param in self.network_nonlinear_corr.parameters()])

            loss = loss_lf + loss_hf + reg_lf + reg_linear_corr + reg_nonlinear_corr # final loss

            loss.backward() # backward pass

            optimizer.step() # update the parameters

        if convert_to_eval_mode:
            for network in networks:
                network.eval() # set network in train mode

    def predict(self, x: torch.Tensor, with_grad: bool = False) -> torch.Tensor:
        """
            Method to predict the output for the given input data

            Parameters
            ----------
            x: torch.Tensor
                a torch tensor representing the input data used for prediction

            with_grad: bool
                flag to determine whether to use no grad context during prediction.
                The default value is False, i.e., predictions will be made without
                tracking any gradients. Set this flag to `True` if you want to track
                gradients

            Returns
            -------
            y_pred: torch.Tensor
                a torch tensor representing the predicted output for the given input data
        """

        assert isinstance(x, torch.Tensor), "`x` should be a torch tensor"
        assert x.device == self.x_train_hf.device, "input data should be on the same device as the training data"
        assert isinstance(with_grad, bool), "`with_grad` should be a boolean value"

        x_ndim = x.ndim # number of dimensions in the given input data

        # check input shape and add batch dim if necessary
        if x_ndim == self.x_train_hf.ndim:
            assert x.shape[1:] == self.x_train_hf.shape[1:], "input data should have the same feature size as the training data"
        elif x_ndim == self.x_train_hf.ndim - 1:
            assert x.shape == self.x_train_hf.shape[1:], "input data should have the same feature size as the training data"
            x = x.unsqueeze(0) # add batch dim as 1
        else:
            raise ValueError("input data should be of similar shape as the training data")
        
        # check if network is in train mode
        if self.network_lf.training or self.network_linear_corr.training or self.network_nonlinear_corr.training:
            raise RuntimeError("Network is in train mode, please use the `fit` method to train the network first and then call the `predict` method")

        # transform the training data
        if self.input_transform is not None:
            x = self.input_transform.transform(x)

        # predict
        if with_grad:
            y_lf = self.network_lf(x)

            x_hf = torch.hstack((x, y_lf)) # combined input for correlation networks

            y_linear_corr = self.network_linear_corr(x_hf)

            y_nonlinear_corr = self.network_nonlinear_corr(x_hf)
                
            y_pred = y_linear_corr + y_nonlinear_corr
        else:
            with torch.no_grad():
                y_lf = self.network_lf(x)

                x_hf = torch.hstack((x, y_lf)) # combined input for correlation networks

                y_linear_corr = self.network_linear_corr(x_hf)

                y_nonlinear_corr = self.network_nonlinear_corr(x_hf)
                    
                y_pred = y_linear_corr + y_nonlinear_corr

        if self.output_transform_hf is not None:
            y_pred = self.output_transform_hf.inverse_transform(y_pred)

        # remove batch dim, if it was added
        if x_ndim == self.x_train_hf.ndim - 1:
            y_pred = y_pred.squeeze(0)
            
        return y_pred

import torch
from ..base_model import BaseModel
from ..utils.transformations import Normalize, Standardize

class Polynomial(BaseModel):

    def __init__(
            self,
            xtrain: torch.Tensor,
            ytrain: torch.Tensor,
            order: int,
            input_transform: Normalize | Standardize | None = None, 
            output_transform: Normalize | Standardize | None = None, 
        ):
        """
            Class definition for nth order polynomial model:

                y = f(x) = 1 + a1*x + a2*x^2 + ... + an*x^n

            Note: This model is only for one dimensional problems

            Parameters
            ----------
            xtrain: torch.Tensor
                x training data for fitting the model - size (N,1)

            ytrain: torch.Tensor
                y training data for fitting the model - size (N,1)

            order: int
                order of the polynomial to fit the data

            input_transform: 
                transformation class object to transform the input data before fitting the model.
                By default, no transformation will be applied

            output_transform:
                transformation class object to transform the output data before fitting the model
                By default, no transformation will be applied
        """
        
        # checks
        assert isinstance(xtrain, torch.Tensor) and xtrain.ndim == 2, "xtrain must be a 2D torch tensor"
        assert isinstance(ytrain, torch.Tensor) and ytrain.ndim == 2, "ytrain must be a 2D torch tensor"
        assert xtrain.shape[0] == ytrain.shape[0], "number of samples in xtrain and ytrain must be the same"
        assert xtrain.shape[1] == ytrain.shape[1], "xtrain and ytrain should be of size (N,1), N is the number of samples"
        assert xtrain.device == ytrain.device, "input and output training data must be on same device"
        assert isinstance(order, int) and order > 0, "order must be a positive integer"

        if input_transform is not None:
            assert isinstance(input_transform, Normalize) or isinstance(input_transform, Standardize), "'input_transform' should be an instance of Normalize or Standardize class"
        if output_transform is not None:
            assert isinstance(output_transform, Normalize) or isinstance(output_transform, Standardize), "'output_transform' should be an instance of Normalize or Standardize class"

        # TO DO: implement input and output transform in a better manner

        super().__init__()

        self.order = order
        self.input_transform = input_transform
        self.output_transform = output_transform

        if self.input_transform is None:
            self.xtrain = xtrain
        else:
            self.xtrain = self.input_transform.transform(xtrain)

        if self.output_transform is None:
            self.ytrain = ytrain
        else:
            self.ytrain = self.output_transform.transform(ytrain)

    def fit(self) -> torch.Tensor:
        """
            Method to compute the weights for a given dataset
            
            Returns
            -------
            weights: torch.Tensor
                aray containing weights of the polynomial model
        """

        # initialize some variables
        basis_matrix = torch.zeros((self.xtrain.shape[0], self.order+1)).to(self.xtrain)

        for i in range(self.xtrain.shape[0]):
            for j in range(self.order+1):
                basis_matrix[i,j] = self.xtrain[i,0]**j

        inverse_basis_matrix = torch.linalg.pinv(basis_matrix)

        self.weights = torch.matmul(inverse_basis_matrix, self.ytrain)

        return self.weights.clone()
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
            Method to predict values for a given input

            Parameters
            ----------
            x: torch.Tensor
                input data at which prediction will be made.
                It shoule be 2D array of size (N,1)

            Returns
            -------
            ypred: torch.Tensor
                predicted y values for given input data. It will
                be 2D array of size (N,1)
        """

        assert hasattr(self, "weights"), "model is not fitted, call fit method before predict"
        assert isinstance(x, torch.Tensor) and x.ndim == 2, "given input data must be a 2D torch tensor"
        assert x.shape[1] == 1, "given input data should be of size (N,1)"
        assert x.device == self.xtrain.device, "given input data is not on the same device as training data"

        if self.input_transform is not None:
            x = self.input_transform.transform(x)

        # create basis matrix
        basis_matrix = torch.zeros((x.shape[0], self.order+1)).to(x)

        for i in range(x.shape[0]):
            for j in range(self.order+1):
                basis_matrix[i,j] = x[i,0]**j

        ypred = torch.matmul(basis_matrix, self.weights)

        if self.output_transform is not None:
            ypred = self.output_transform.inverse_transform(ypred)

        return ypred

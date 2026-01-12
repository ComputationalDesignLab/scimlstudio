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

            Note: This model is only for 1D data

            Parameters
            ----------
            xtrain: torch.Tensor
                x training data fitting the model

            ytrain: torch.Tensor
                y training data fitting the model

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
        assert isinstance(xtrain, torch.Tensor) and xtrain.ndim == 1, "xtrain must be a 1D torch.Tensor"
        assert isinstance(ytrain, torch.Tensor) and ytrain.ndim == 1, "ytrain must be a 1D torch.Tensor"
        assert xtrain.shape[0] == ytrain.shape[0], "number of samples in xtrain and ytrain must be the same"
        assert xtrain.device == ytrain.device, "input and output training data must be on same device"
        assert isinstance(order, int) and order > 0, "order must be a positive integer"
        if input_transform is not None:
            assert isinstance(input_transform, Normalize) or isinstance(input_transform, Standardize), "'input_transform' should be an instance of Normalize or Standardize class"
        if output_transform is not None:
            assert isinstance(output_transform, Normalize) or isinstance(output_transform, Standardize), "'output_transform' should be an instance of Normalize or Standardize class"

        # TO DO: implement input and output transform in a better manner

        super().__init__()

        self.xtrain = xtrain
        self.ytrain = ytrain
        self.order = order
        self.input_transform = input_transform
        self.output_transform = output_transform

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
                basis_matrix[i,j] = self.xtrain[i]**j

        inverse_basis_matrix = torch.linalg.pinv(basis_matrix)

        self.weights = torch.matmul(inverse_basis_matrix, self.ytrain)

        return self.weights.clone()
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
            Method to predict values for a given input

            Parameters
            ----------
            x: torch.Tensor
                input data at which prediction will be made

            Returns
            -------
            ypred: torch.Tensor
                predicted y values for given input data
        """

        assert hasattr(self, "weights"), "model is not fitted, call fit method before predict"
        assert isinstance(x, torch.Tensor) and x.ndim == 1, "given input data must be a 1D torch.Tensor"
        assert x.device == self.xtrain.device, "given input data is not on the same device as training data"

        # create basis matrix
        basis_matrix = torch.zeros((x.shape[0], self.order+1)).to(x)

        for i in range(x.shape[0]):
            for j in range(self.order+1):
                basis_matrix[i,j] = x[i]**j

        ypred = torch.matmul(basis_matrix, self.weights)

        return ypred

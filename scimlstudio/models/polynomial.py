import torch
from ..base_model import BaseModel
from ..utils.normalize import Normalize
from ..utils.standardize import Standardize

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

        # initialize weights
        self.weights = torch.zeros((self.xtrain.shape[0],self.order+1), requires_grad=False).to(self.xtrain)

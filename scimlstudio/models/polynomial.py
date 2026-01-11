import torch
from ..base_model import BaseModel

class Polynomial(BaseModel):

    def __init__(self, xtrain, ytrain, order):
        """
            Class definition for nth order polynomial model:

                y = f(x) = 1 + a1*x + a2*x^2 + ... + an*x^n

            Parameters
            ----------
            xtrain: torch.Tensor
                x training data fitting the model

            ytrain: torch.Tensor
                y training data fitting the model

            order: int
                order of the polynomial to fit the data

            input_transform: callable, optional
                function to transform the input data before fitting the model

            
        """
        
        # checks
        assert isinstance(xtrain, torch.Tensor) and xtrain.ndim == 2, "xtrain must be a 2D torch.Tensor"
        assert isinstance(ytrain, torch.Tensor) and ytrain.ndim == 2, "ytrain must be a 2D torch.Tensor"
        assert xtrain.shape[0] == ytrain.shape[0], "number of samples in xtrain and ytrain must be the same"
        assert isinstance(order, int) and order > 0, "order must be a positive integer"

        super().__init__()

        self.xtrain = xtrain


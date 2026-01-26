import torch
from ..base_models.gp_base_model import GPBaseModel
from ..utils.transformations import Standardize, Normalize
from gyptorch.models import ExactGP

class SingleOutputGP(ExactGP, GPBaseModel):

        def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor, likelihood, mean_module, covar_module,
                     input_transform: Normalize | Standardize | None = None, output_transform: Normalize | Standardize | None = None):

                """
                        Class definition for single output GP models with exact inference

                        Parameters
                        ----------
                        x_train: torch.Tensor
                                Input training data for the GP model
                        y_train: torch.Tensor
                                Output training data for the GP model
                        likelihood:
                                Likelihood for the GP model
                        mean_module:
                                Mean function for the GP model
                        covar_module: 
                                Covariance function for the GP model
                        input_transform: Normalize or Standardize or None
                                Data scaling class for the inputs of the GP model
                        output_transform: Normalize or Standardize or None
                                Data scaling class for the outputs of the GP model
                
                """

                super(SingleOutputGP, self).__init__(x_train, y_train, likelihood)

                # Assigning the likelihood, mean function and covariance function
                self.likelihood = likelihood
                self.mean_module = mean_module
                self.covar_module = covar_module

                # Assigning the data transforms
                self.input_transform = input_transform
                self.output_transform = output_transform

        def transform_inputs(self):
                pass

        def transform_outputs(self):
                pass

        def fit(self):
                pass

        def predict(self):
                pass

        def posterior(self):
                pass
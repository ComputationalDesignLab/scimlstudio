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

                """
                        Method to transform the inputs based on the provided transform

                        Returns
                        -------
                        x_transformed: torch.Tensor
                                Transformed input data based on provided transform
                """

                if self.input_transform is not None:
                        self.input_transform = self.input_transform(self.x_train)
                        x_transformed = self.input_transform.transform(self.x_train)
                        return x_transformed
                
                else:
                        return self.x_train

        def transform_outputs(self):
                
                """
                        Method to transform the outputs based on the provided transform

                        Returns
                        -------
                        y_transformed: torch.Tensor
                                Transformed output data based on provided transform
                """

                if self.output_transform is not None:
                        self.output_transform = self.output_transform(self.y_train)
                        y_transformed = self.output_transform.transform(self.y_train)
                        return y_transformed
                
                else:
                        return self.y_train

        def fit(self, training_iterations: int, mll, learning_rate: float):
                """
                        Method to fit the GP model to the given data

                        Parameters
                        ----------
                        training_iterations: int
                                Number of iterations for the fitting process
                        mll:
                                Marginal log likelihood used to fit the GP model 
                                "Loss" function for the GP models
                        learning_rate:
                                The value of the learning rate for the optimization
                """

                # Putting the model in train mode
                self.train()
                self.likelihood.train()

                # Setting the optimizer
                optim = torch.optim.Adam(self.parameters(), lr=learning_rate)

                # Running the optimization loop to fit the model
                for i in range(training_iterations):

                        optim.zero_grad()
                        model_output = self(self.x_train)
                        loss_function = -mll(model_output, self.y_train)
                        loss_function.backward()
                        optim.step()

        def predict(self):
                pass

        def posterior(self):
                pass
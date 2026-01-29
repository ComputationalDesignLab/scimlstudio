import torch
import gpytorch
from ..base_models.gp_base_model import GPBaseModel
from ..utils.transformations import Standardize, Normalize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior, get_gaussian_likelihood_with_lognormal_prior
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal

class SingleOutputGP(ExactGP, GPBaseModel):

        def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor, likelihood_input = None, mean_module = None, covar_module = None,
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

                # Assigning the likelihood
                if likelihood_input is None:
                        likelihood = get_gaussian_likelihood_with_lognormal_prior()
                else:
                        likelihood = likelihood_input

                # Initiliazing the parent class
                super(SingleOutputGP, self).__init__(x_train, y_train, likelihood)
                self.likelihood = likelihood

                # Assigning the mean function
                if mean_module is None:
                        self.mean_module = ConstantMean()
                else:
                        self.mean_module = mean_module
                
                # Assigning the covariance function
                if covar_module is None:
                        self.covar_module = get_covar_module_with_dim_scaled_prior(ard_num_dims=x_train.shape[-1])
                else:
                        self.covar_module = covar_module

                # Assigning the data transforms
                self.input_transform = input_transform
                self.output_transform = output_transform

                self.x_train = self.transform_inputs()
                self.y_train = self.transform_outputs()

        def transform_inputs(self) -> torch.Tensor:

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

        def transform_outputs(self) -> torch.Tensor:
                
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

        def fit(self, training_iterations: int, mll: gpytorch.mlls.ExactMarginalLogLikelihood, optimizer: torch.optim.Optimizer, 
                verbose: bool = False):
                
                """
                        Method to fit the GP model to the given data

                        Parameters
                        ----------
                        training_iterations: int
                                Number of iterations for the fitting process
                        mll: gpytorch.mlls.ExactMarginalLogLikelihood
                                Marginal log likelihood used to fit the GP model 
                                "Loss" function for the GP models
                        optimizer: torch.optim.Optimizer
                                Optimizer for finding optimal hyperparameters
                        verbose: bool
                                Boolean flag of whether to print the progress of training
                """

                # Putting the model in train mode
                self.train()
                self.likelihood.train()

                # Setting the optimizer
                optim = optimizer

                # Running the optimization loop to fit the model
                for iter in range(training_iterations):

                        optim.zero_grad() # Zeroing gradients
                        model_output = self(self.x_train) # Output of the GP model
                        loss_function = -mll(model_output, self.y_train) # Calculating MLL as the loss
                        loss_function.backward() # Backward pass to calculate gradients
                        optim.step() # Making an optimizer step
                        if verbose:
                                print(f"Iteration {iter}/{training_iterations}: Marginal log likelihood {loss_function.item()}")

        def forward(self, x: torch.Tensor) -> MultivariateNormal:
                
                """
                        Forward method for the model
                        
                        Parameters
                        ----------
                        x: torch.Tensor
                                Inputs to the model where predictions are desired

                        Returns
                        -------
                        dist: MultivariateNormal
                                MultivariateNormal prediction from the GP model
                """

                mean = self.mean_module(x) # Evaluate the mean function
                covariance = self.covar_module(x) # Evaluate the covariance function
                dist = MultivariateNormal(mean, covariance) # Calculate the multivariate normal distribution

                return dist

        def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                
                """
                        Method for calculating predictions on given input data

                        Parameters
                        ----------
                        x: torch.Tensor
                                Input data for calculating the predictions

                        Returns
                        -------
                        mean_values: torch.Tensor
                                Mean predictions from the GP model
                        std_values: torch.Tensor
                                Standard deviations from the GP model
                """
                # Putting the model into eval mode
                self.eval()
                self.likelihood.eval()
                if self.input_transform is not None:
                        x = self.input_transform.transform(x)

                # Do not track gradients and use gpytorch setting to calculate faster prediction
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        predictions = self.likelihood(self(x))

                # Mean values of the prediction
                mean_values = predictions.mean

                # Confidence region - provides 2 times standard deviations 
                lower_values, _ = predictions.confidence_region()
                std_values = 0.5 * (mean_values - lower_values)

                if self.output_transform is not None:
                        mean_values, std_values = self.output_transform.inverse_transform(mean_values), self.output_transform.inverse_transform(std_values)

                return mean_values, std_values


        def posterior(self, x: torch.Tensor) -> GPyTorchPosterior:
                
                """
                        Method for calculating the posterior distribution of the GP model

                        Parameters
                        ----------
                        x: torch.Tensor
                                Input data where posterior distribution must be calculated

                        Returns
                        -------
                        posterior: GPyTorchPosterior
                                Posterior distribution object for a gpytorch model
                """

                # Putting the model into eval mode
                self.eval()
                self.likelihood.eval()

                # Determining the posterior
                dist = self.likelihood(self(x))
                posterior = GPyTorchPosterior(mvn=dist)
                return posterior

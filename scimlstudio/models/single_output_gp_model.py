import torch
import gpytorch
import math
from ..base_models.gp_base_model import GPBaseModel
from ..utils.transformations import Standardize, Normalize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.utils.gpytorch_modules import get_gaussian_likelihood_with_lognormal_prior
from gpytorch.means import ConstantMean, Mean
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, Kernel, RBFKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import LogNormalPrior

def get_dim_scaled_covar_module(base_covar_module: Kernel, ard_num_dims: int, batch_shape: torch.Size | None = None) -> ScaleKernel:
    """
        Customized version of get_covar_module_with_dim_scaled_prior from BoTorch package.
        Main customization is to allow the usage of any kernel with the dimension scaled log normal prior on the lengthscale.

        Parameters
        ----------
        base_covar_module: Kernel
                The kernel function that acts as the base and upon which the prior and scaling is performed
        ard_num_dims: int
                Number of dimensions in the input data with each input dimensions having a separate lengthscale
        batch_shape: torch.Size or None
                Batch shape of the input data

        Returns
        -------
        scaled_kernel: ScaleKernel
                Kernel function with the prior applied and scaled using the ScaleKernel
    """

    lengthscale_prior = LogNormalPrior(loc=math.sqrt(2) + math.log(ard_num_dims) * 0.5, scale=math.sqrt(3))
    kernel = base_covar_module(ard_num_dims=ard_num_dims, batch_shape=batch_shape, lengthscale_prior=lengthscale_prior,
                               lengthscale_constraint=GreaterThan(2.5e-2, transform=None, initial_value=lengthscale_prior.mode))
    scaled_kernel = ScaleKernel(kernel)
    return scaled_kernel

class SingleOutputGP(ExactGP, GPBaseModel):

        def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor, likelihood_module: Likelihood = None, mean_module: Mean = None, covar_module: Kernel = None,
                     input_transform: Normalize | Standardize | None = None, output_transform: Normalize | Standardize | None = None,
                     noiseless: bool = True, use_dim_scaling: bool = True):

                """
                        Class definition for single output GP models with exact inference

                        Parameters
                        ----------
                        x_train: torch.Tensor
                                Input training data for the GP model in a 2D tensor
                        y_train: torch.Tensor
                                Output training data for the GP model in a 1D tensor
                        likelihood_module: Likelihood
                                Likelihood for the GP model
                        mean_module: Mean
                                Mean function for the GP model
                        covar_module: Kernel
                                Covariance function for the GP model
                                If providing a covar_module, it should be a Kernel class
                                and not a specific instance of the class
                        input_transform: Normalize or Standardize or None
                                Data scaling class for the inputs of the GP model
                        output_transform: Normalize or Standardize or None
                                Data scaling class for the outputs of the GP model
                        noiseless: bool
                                Boolean flag to indicate whether data is noiseless
                        use_dim_scaling: bool
                                Boolean flag to indicate whether the kernel function
                                should have a scaled prior based on input dimensionality
                
                """
                # Some checks
                assert isinstance(x_train, torch.Tensor) and x_train.ndim == 2, "xtrain must be a 2D tensor array"
                assert isinstance(y_train, torch.Tensor) and y_train.ndim == 2, "ytrain must be a 2D tensor array"
                assert x_train.shape[0] == y_train.shape[0], "number of samples in input and output training data must be the same"
                assert x_train.device == y_train.device, "input and output training data must be on the same device"
                assert isinstance(noiseless, bool), "noiseless must be a Boolean value"
                assert isinstance(use_dim_scaling, bool), "use_dim_scaling must be a Boolean value"

                # Assigning the likelihood
                likelihood = get_gaussian_likelihood_with_lognormal_prior() if likelihood_module is None else likelihood_module

                # Initiliazing the parent class
                xtrain = self.transform_inputs(x_train, input_transform)
                ytrain = self.transform_outputs(y_train, output_transform)
                super(SingleOutputGP, self).__init__(xtrain, ytrain.reshape(-1,), likelihood)
                # TO DO: see if the noiseless GP can be implemented in a better way
                if noiseless:
                        self.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(1e-12))
                        self.likelihood.noise = 1e-12 # Need to tell likelihood that noise is zero

                self.y_train = ytrain
                self.x_train = xtrain

                # Assigning the mean function
                self.mean_module = ConstantMean() if mean_module is None else mean_module
                
                # Assigning the covariance function 
                covar_module = RBFKernel if covar_module is None else covar_module
                if use_dim_scaling:
                        self.covar_module = get_dim_scaled_covar_module(base_covar_module=covar_module, ard_num_dims=xtrain.shape[-1])
                else:
                        self.covar_module = ScaleKernel(covar_module(ard_num_dims=xtrain.shape[-1]))

                # Setting the model to GPU if the data provided is also on GPU
                if xtrain.device.type != 'cpu':
                        self.cuda()
                        self.likelihood.cuda()

        def transform_inputs(self, x: torch.Tensor, input_transform: Normalize | Standardize | None) -> torch.Tensor:

                """
                        Method to transform the inputs based on the provided transform

                        Parameters
                        ----------
                        x: torch.Tensor
                                Input data in a 2D tensor that must be transformed
                        input_transform: Normalize or Standardize or None
                                Data scaling class for the inputs of the GP model

                        Returns
                        -------
                        x_transformed: torch.Tensor
                                Transformed input data based on provided transform
                                If transform is None, original data is returned
                """
                # TO DO: see if transform_inputs and outputs can be implemented in a better manner
                assert isinstance(x, torch.Tensor) and x.ndim == 2, "x must be a 2D tensor array"

                if input_transform is not None:
                        assert isinstance(input_transform(x), Normalize) or isinstance(input_transform(x), Standardize), "input transform must be an instance of Normalize or Standardize class"
                        self.input_transform = input_transform(x)
                        x_transformed = self.input_transform.transform(x)
                        return x_transformed
                
                else:
                        self.input_transform = None
                        x_transformed = x
                        return x_transformed

        def transform_outputs(self, y: torch.Tensor, output_transform: Normalize | Standardize | None) -> torch.Tensor:
                
                """
                        Method to transform the outputs based on the provided transform

                        Parameters
                        ----------
                        y: torch.Tensor
                                Output data in a 2D tensor that must be transformed
                        output_transform: Normalize or Standardize or None
                                Data scaling class for the outputs of the GP model

                        Returns
                        -------
                        y_transformed: torch.Tensor
                                Transformed output data based on provided transform
                                If transform is None, original data is returned
                """

                assert isinstance(y, torch.Tensor) and y.ndim == 2, "x must be a 2D tensor array"

                if output_transform is not None:
                        assert isinstance(output_transform(y), Normalize) or isinstance(output_transform(y), Standardize), "output transform must be an instance of Normalize or Standardize class"
                        self.output_transform = output_transform(y)
                        y_transformed = self.output_transform.transform(y)
                        return y_transformed
                
                else:
                        self.output_transform = None
                        y_transformed = y
                        return y_transformed

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

                # Flattened version of training outputs
                y_train_flattened = self.y_train.reshape(-1,)

                # Running the optimization loop to fit the model
                for iter in range(training_iterations):

                        optim.zero_grad() # Zeroing gradients
                        model_output = self(self.x_train) # Output of the GP model
                        loss_function = -mll(model_output, y_train_flattened) # Calculating MLL as the loss
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

                assert isinstance(x, torch.Tensor) and x.ndim == 2, "x must be a 2D tensor array"
                assert x.device == self.x_train.device, "data to be transformed must be on the same device as training data"

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
                assert isinstance(x, torch.Tensor) and x.ndim == 2, "x must be a 2D tensor array"
                assert x.device == self.x_train.device, "data to be transformed must be on the same device as training data"

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

                # Standard deviation of the prediction
                std_values = predictions.stddev

                if self.output_transform is not None:
                        mean_values = self.output_transform.inverse_transform(mean_values.reshape(-1,1))
                        if isinstance(self.output_transform, Standardize):
                                std_values = self.output_transform.std * std_values.reshape(-1,1)
                        elif isinstance(self.output_transform, Normalize):
                                std_values = (self.output_transform.max_feature_vector - self.output_transform.min_feature_vector) * std_values.reshape(-1,1)
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
                # TO DO: implement a posterior transform to account for data transforms automatically
                # Putting the model into eval mode
                self.eval()
                self.likelihood.eval()

                # Determining the posterior
                dist = self.likelihood(self(x))
                posterior = GPyTorchPosterior(dist)
                return posterior

import warnings
warnings.filterwarnings("ignore")
import unittest, torch, math
from gpytorch.kernels import RBFKernel, MaternKernel, RQKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from scimlstudio.utils import evaluate_scalar
from scimlstudio.models import SingleOutputGP
from scimlstudio.utils import Normalize, Standardize
from pyDOE3 import lhs, halton_sequence

# There will be a few warnings from gpytorch as predictions will be made on training data
# but these can be ignored as that is the intent in the test cases

# Defining the device and data type
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.float64}

class TestSingleOutputGP(unittest.TestCase):

    """
        Class defining the test cases for the single output GP model
    """
    
    def test_gp(self):

        # Generating some data for the test cases
        # Function used for the test cases is a sinusoidal function (https://www.sfu.ca/~ssurjano/curretal88sin.html)
        x_train = torch.linspace(0, 1, 5, **tkwargs)
        y_train = torch.sin(2*math.pi*(x_train - 0.1))

        x_test = torch.linspace(0, 1, 100, **tkwargs)
        y_test = torch.sin(2*math.pi*(x_test - 0.1))

        # Creating and fitting GP models with different kernel functions
        for kernel in [RBFKernel, MaternKernel, RQKernel]:
            gp = SingleOutputGP(x_train=x_train.reshape(-1,1), y_train=y_train.reshape(-1,1), covar_module=kernel)
            
            # Defining a few things to train the model
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp) # loss function 
            optimizer = torch.optim.Adam(gp.parameters(), lr=0.01) # optimizer

            # Training the model
            gp.fit(training_iterations=1000, mll=mll, optimizer=optimizer)

            train_pred, train_std = gp.predict(x_train.reshape(-1,1))
            test_pred, test_std = gp.predict(x_test.reshape(-1,1))
            r2_value = evaluate_scalar(test_pred.reshape(-1,), y_test, "r2")

            torch.testing.assert_close(train_pred.reshape(-1,), y_train, rtol=0, atol=1e-6, check_device=True, check_dtype=True) # interpolation check
            torch.testing.assert_close(train_std.reshape(-1,), torch.zeros_like(train_std.reshape(-1,), **tkwargs), rtol=0, atol=5e-4, check_device=True, check_dtype=True) # interpolation check
            assert torch.all(test_std.reshape(-1,) > 0.0) # standard deviation must be positive
            assert round(r2_value, 6) < 1 

        # Creating and fitting GP models with normalize transformations
        for kernel in [RBFKernel, MaternKernel, RQKernel]:
            gp = SingleOutputGP(x_train=x_train.reshape(-1,1), y_train=y_train.reshape(-1,1), covar_module=kernel, input_transform=Normalize, output_transform=Normalize)

            # Defining a few things to train the model
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp) # loss function 
            optimizer = torch.optim.Adam(gp.parameters(), lr=0.01) # optimizer

            # Training the model
            gp.fit(training_iterations=1000, mll=mll, optimizer=optimizer)

            train_pred, train_std = gp.predict(x_train.reshape(-1,1))
            test_pred, test_std = gp.predict(x_test.reshape(-1,1))
            r2_value = evaluate_scalar(test_pred.reshape(-1,), y_test, "r2")

            torch.testing.assert_close(train_pred.reshape(-1,), y_train, rtol=0, atol=1e-6, check_device=True, check_dtype=True) # interpolation check
            torch.testing.assert_close(train_std.reshape(-1,), torch.zeros_like(train_std.reshape(-1,), **tkwargs), rtol=0, atol=5e-4, check_device=True, check_dtype=True) # interpolation check
            assert torch.all(test_std.reshape(-1,) > 0.0) # standard deviation must be positive
            assert round(r2_value, 6) < 1 

    def test_gp_5D(self):

        # Generating some data for the test cases
        # Function used for the test cases is the Friedman function (https://www.sfu.ca/~ssurjano/fried.html)
        x_train = torch.tensor(halton_sequence(num_points=15, dimension=5), **tkwargs)
        y_train = 10*torch.sin(math.pi*x_train[:,0]*x_train[:,1]) + 20 * (x_train[:,2] - 0.5) ** 2 + 10 * x_train[:,3] + 5 * x_train[:,4]

        x_test = torch.tensor(lhs(n=5, samples=100, criterion='cm', iterations=100, seed=10), **tkwargs)
        y_test = 10*torch.sin(math.pi*x_test[:,0]*x_test[:,1]) + 20 * (x_test[:,2] - 0.5) ** 2 + 10 * x_test[:,3] + 5 * x_test[:,4]

        # Creating and fitting GP models with different kernel functions
        for kernel in [RBFKernel, MaternKernel, RQKernel]:
            gp = SingleOutputGP(x_train=x_train, y_train=y_train.reshape(-1,1), covar_module=kernel)

            # Defining a few things to train the model
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp) # loss function 
            optimizer = torch.optim.Adam(gp.parameters(), lr=0.01) # optimizer

            # Training the model
            gp.fit(training_iterations=1000, mll=mll, optimizer=optimizer)

            train_pred, train_std = gp.predict(x_train)
            test_pred, test_std = gp.predict(x_test)
            r2_value = evaluate_scalar(test_pred.reshape(-1,), y_test, "r2")

            torch.testing.assert_close(train_pred.reshape(-1,), y_train, rtol=0, atol=1e-6, check_device=True, check_dtype=True) # interpolation check
            torch.testing.assert_close(train_std.reshape(-1,), torch.zeros_like(train_std.reshape(-1,), **tkwargs), rtol=0, atol=5e-4, check_device=True, check_dtype=True) # interpolation check
            assert torch.all(test_std.reshape(-1,) > 0.0) # standard deviation must be positive
            assert round(r2_value, 6) < 1 

        # Creating and fitting GP models with standardize transformations
        for kernel in [RBFKernel, MaternKernel, RQKernel]:
            gp = SingleOutputGP(x_train=x_train, y_train=y_train.reshape(-1,1), covar_module=kernel, input_transform=Standardize, output_transform=Standardize)

            # Defining a few things to train the model
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp) # loss function 
            optimizer = torch.optim.Adam(gp.parameters(), lr=0.01) # optimizer

            # Training the model
            gp.fit(training_iterations=1000, mll=mll, optimizer=optimizer)

            train_pred, train_std = gp.predict(x_train)
            test_pred, test_std = gp.predict(x_test)
            r2_value = evaluate_scalar(test_pred.reshape(-1,), y_test, "r2")

            torch.testing.assert_close(train_pred.reshape(-1,), y_train, rtol=0, atol=1e-6, check_device=True, check_dtype=True) # interpolation check
            torch.testing.assert_close(train_std.reshape(-1,), torch.zeros_like(train_std.reshape(-1,), **tkwargs), rtol=0, atol=5e-4, check_device=True, check_dtype=True) # interpolation check
            assert torch.all(test_std.reshape(-1,) > 0.0) # standard deviation must be positive
            assert round(r2_value, 6) < 1 

    def test_inputs(self):

        # Generate dummy data
        x_random = torch.rand(15)
        y_random = torch.rand(15)

        with self.assertRaises(Exception):
            _ = SingleOutputGP(x_train=x_random.reshape(-1,1), y_train=y_random)

        with self.assertRaises(Exception):
            _ = SingleOutputGP(x_train=x_random, y_train=y_random.reshape(-1,1))

        with self.assertRaises(Exception):
            _ = SingleOutputGP(x_train=x_random.reshape(-1,1), y_train=y_random.reshape(-1,1), noiseless=0.0, use_dim_scaling=True)

        with self.assertRaises(Exception):
            _ = SingleOutputGP(x_train=x_random.reshape(-1,1), y_train=y_random.reshape(-1,1), noiseless=True, use_dim_scaling=1.0)

        _ = SingleOutputGP(x_train=x_random.reshape(-1,1), y_train=y_random.reshape(-1,1), noiseless=True, use_dim_scaling=True)


if __name__ == '__main__':
    unittest.main()


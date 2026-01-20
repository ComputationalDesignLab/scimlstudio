import unittest, torch, math
from scimlstudio.utils import evaluate_scalar
from scimlstudio.models import RBF
from scimlstudio.utils import Normalize, Standardize
from pyDOE3 import lhs, halton_sequence

# Defining the device and data type
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.float64}

class TestRBF(unittest.TestCase):

    """
        Class defining the test cases for the RBF model
    """
    
    def test_rbf(self):

        # Generating some data for the test cases
        # Function used for the test cases is a sinusoidal function (https://www.sfu.ca/~ssurjano/curretal88sin.html)
        x_train = torch.linspace(0, 1, 10, **tkwargs)
        y_train = torch.sin(2*math.pi*(x_train - 0.1))

        x_test = torch.linspace(0, 1, 100, **tkwargs)
        y_test = torch.sin(2*math.pi*(x_test - 0.1))

        # Creating and fitting RBF models
        for basis in ["gaussian", "multiquadric", "inverse"]:
            rbf = RBF(x_train.reshape(-1,1), y_train.reshape(-1,1), sigma = 0.1, basis = basis)
            _, basis_matrix = rbf.fit()
            train_pred = rbf.predict(x_train.reshape(-1,1))
            test_pred = rbf.predict(x_test.reshape(-1,1))
            r2_value = evaluate_scalar(test_pred.reshape(-1,), y_test, "r2")

            torch.testing.assert_close(train_pred.reshape(-1,), y_train, rtol=0, atol=1e-6, check_device=True, check_dtype=True) # interpolation check
            assert basis_matrix.ndim == 2 # basis matrix must be 2D
            if basis == "gaussian":
                assert (torch.diag(basis_matrix) == 1).all() # diagonal of basis matrix must be all 1s if basis is gaussian
            assert round(r2_value, 6) < 1 

        # Creating and fitting RBF models with normalize transformations
        input_transform = Normalize(x_train.reshape(-1,1))
        output_transform = Normalize(y_train.reshape(-1,1))
        for basis in ["gaussian", "multiquadric", "inverse"]:
            rbf = RBF(x_train.reshape(-1,1), y_train.reshape(-1,1), sigma = 0.1, basis = basis, input_transform=input_transform, output_transform=output_transform)
            _, basis_matrix = rbf.fit()
            train_pred = rbf.predict(x_train.reshape(-1,1))
            test_pred = rbf.predict(x_test.reshape(-1,1))
            r2_value = evaluate_scalar(test_pred.reshape(-1,), y_test, "r2")

            torch.testing.assert_close(train_pred.reshape(-1,), y_train, rtol=0, atol=1e-6, check_device=True, check_dtype=True) # interpolation check
            assert basis_matrix.ndim == 2 # basis matrix must be 2D
            if basis == "gaussian":
                assert (torch.diag(basis_matrix) == 1).all() # diagonal of basis matrix must be all 1s if basis is gaussian
            assert round(r2_value, 6) < 1 

    def test_rbf_5D(self):

        # Generating some data for the test cases
        # Function used for the test cases is the Friedman function (https://www.sfu.ca/~ssurjano/fried.html)
        x_train = torch.tensor(halton_sequence(num_points=15, dimension=5), **tkwargs)
        y_train = 10*torch.sin(math.pi*x_train[:,0]*x_train[:,1]) + 20 * (x_train[:,2] - 0.5) ** 2 + 10 * x_train[:,3] + 5 * x_train[:,4]

        x_test = torch.tensor(lhs(n=5, samples=100, criterion='cm', iterations=100, seed=10), **tkwargs)
        y_test = 10*torch.sin(math.pi*x_test[:,0]*x_test[:,1]) + 20 * (x_test[:,2] - 0.5) ** 2 + 10 * x_test[:,3] + 5 * x_test[:,4]

        # Creating and fitting RBF models
        for basis in ["gaussian", "multiquadric", "inverse"]:
            rbf = RBF(x_train, y_train.reshape(-1,1), sigma = 0.1, basis = basis)
            _, basis_matrix = rbf.fit()
            train_pred = rbf.predict(x_train)
            test_pred = rbf.predict(x_test)
            r2_value = evaluate_scalar(test_pred.reshape(-1,), y_test, "r2")

            torch.testing.assert_close(train_pred.reshape(-1,), y_train, rtol=0, atol=1e-6, check_device=True, check_dtype=True) # interpolation check
            assert basis_matrix.ndim == 2 # basis matrix must be 2D
            if basis == "gaussian":
                assert (torch.diag(basis_matrix) == 1).all() # diagonal of basis matrix must be all 1s if basis is gaussian
            assert round(r2_value, 6) < 1 

        # Creating and fitting RBF models with standardize transformations
        input_transform = Standardize(x_train)
        output_transform = Standardize(y_train.reshape(-1,1))
        for basis in ["gaussian", "multiquadric", "inverse"]:
            rbf = RBF(x_train, y_train.reshape(-1,1), sigma = 0.1, basis = basis, input_transform=input_transform, output_transform=output_transform)
            _, basis_matrix = rbf.fit()
            train_pred = rbf.predict(x_train)
            test_pred = rbf.predict(x_test)
            r2_value = evaluate_scalar(test_pred.reshape(-1,), y_test, "r2")

            torch.testing.assert_close(train_pred.reshape(-1,), y_train, rtol=0, atol=1e-6, check_device=True, check_dtype=True) # interpolation check
            assert basis_matrix.ndim == 2 # basis matrix must be 2D
            if basis == "gaussian":
                assert (torch.diag(basis_matrix) == 1).all() # diagonal of basis matrix must be all 1s if basis is gaussian
            assert round(r2_value, 6) < 1

    def test_inputs(self):

        # Generate dummy data
        x_random = torch.rand(15)
        y_random = torch.rand(15)

        with self.assertRaises(Exception):
            _ = RBF(x_random.reshape(-1,1), y_random, sigma = 0.1)

        with self.assertRaises(Exception):
            _ = RBF(x_random, y_random.reshape(-1,1), sigma = 0.1)

        with self.assertRaises(Exception):
            _ = RBF(x_random.reshape(-1,1), y_random.reshape(-1,1), sigma = -0.1)

        with self.assertRaises(Exception):
            _ = RBF(x_random.reshape(-1,1), y_random.reshape(-1,1), sigma = 0.1, basis="linear")

        _ = RBF(x_random.reshape(-1,1), y_random.reshape(-1,1), sigma = 0.1, basis="inverse")

if __name__ == '__main__':
    unittest.main()


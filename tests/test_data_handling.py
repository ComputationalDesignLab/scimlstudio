import unittest, torch
from scimlstudio.utils.normalize import Normalize
from scimlstudio.utils.standardize import Standardize

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64

class TestDataHandling(unittest.TestCase):
    """
        Class defining test cases for data handling classes
    """

    def test_normalization(self):

        # Generate data
        num_samples = 10
        X = torch.rand((num_samples,4)).to(device=device, dtype=dtype)
        lb = torch.tensor([-0.2, 0.0, 0.3, -1.0])
        ub = torch.tensor([0.7, 1.0, 1.2, -0.2])

        # initialize the class
        xtransform = Normalize(X, lower_bound=lb, upper_bound=ub)

        # scale the data
        X_normalized = xtransform.transform(X)

        # check
        torch.testing.assert_close(X, xtransform.inverse_transform(X_normalized), rtol=0, atol=1e-6, check_device=True, check_dtype=True)
        torch.testing.assert_close(torch.min(X_normalized, axis=0)[0], lb, rtol=0, atol=1e-6, check_device=False, check_dtype=False)
        torch.testing.assert_close(torch.max(X_normalized, axis=0)[0], ub, rtol=0, atol=1e-6, check_device=False, check_dtype=False)
        assert X.device == X_normalized.device
        assert X.dtype == X_normalized.dtype

    def test_standardization(self):

        # Construct a random tensor in place of a data set
        n_rows = 20
        n_col = 5
        Y = torch.rand((n_rows, n_col)).to(device=device, dtype=dtype)

        # Directly calculating the mean and standard deviation of random tensor
        mean = torch.mean(Y, dim=0)
        std = torch.clamp(torch.std(Y, dim=0), 1e-8)

        # Initializing the standardize class
        ytransform = Standardize(Y, mean=mean, std=std)

        # Transforming the data
        Y_standardized = ytransform.transform(Y)
        Y_reconstruction = ytransform.inverse_transform(Y_standardized)

        # Checks for standardization
        torch.testing.assert_close(Y_reconstruction, Y, rtol=0, atol=1e-6, check_device=True, check_dtype=True)
        torch.testing.assert_close(torch.mean(Y_standardized, dim=0), torch.zeros(n_col), rtol=0, atol=1e-6, check_device=False, check_dtype=False)
        torch.testing.assert_close(torch.std(Y_standardized, dim=0), torch.ones(n_col), rtol=0, atol=1e-6, check_device=False, check_dtype=False)

        # Few assertions
        assert Y.device == Y_standardized.device
        assert Y.dtype == Y_standardized.dtype
        
if __name__ == '__main__':
    unittest.main()

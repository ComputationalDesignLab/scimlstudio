import unittest, torch
from scimlstudio.utils.normalize import Normalize

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64

class TestNormalize(unittest.TestCase):
    """
        Class defining test cases for normalization class
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

if __name__ == '__main__':
    unittest.main()

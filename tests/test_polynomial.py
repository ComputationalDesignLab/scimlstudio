import unittest, torch
from scimlstudio.models import Polynomial
from scimlstudio.utils import evaluate_scalar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
tkwargs = {"device": device, "dtype": dtype}

class TestPolynomial(unittest.TestCase):
    """
        Class defining test cases for polynomial model
    """

    def test_(self):

        # test and training data
        xtrain = torch.linspace(0, 1, 10, **tkwargs).reshape(-1,1)
        ytrain = (6*xtrain - 2)**2 * torch.sin(12*xtrain - 4)

        xtest = torch.linspace(0, 1, 100, **tkwargs).reshape(-1,1)
        ytest = (6*xtest - 2)**2 * torch.sin(12*xtest - 4)

        # create model
        model = Polynomial(xtrain, ytrain, order=5)
        model.fit()
        ypred = model.predict(xtrain)
        r2 = evaluate_scalar(ypred.reshape(-1,), ytrain.reshape(-1,), "r2")

        assert round(r2, 6) < 1

        # create model
        model = Polynomial(xtrain, ytrain, order=10)
        model.fit()
        ypred = model.predict(xtrain)
        rmse = evaluate_scalar(ypred.reshape(-1,), ytrain.reshape(-1,), "rmse")
        nrmse = evaluate_scalar(ypred.reshape(-1,), ytrain.reshape(-1,), "nrmse")
        r2 = evaluate_scalar(ypred.reshape(-1,), ytrain.reshape(-1,), "r2")

        assert rmse < 1e-8
        assert nrmse < 1e-8
        assert round(r2, 6) == 1

    def test_training_data_shape(self):
        """
            Method to test the input parameters of the class
        """

        # dummy training data
        xtrain = torch.rand(10)
        ytrain = torch.rand(10)

        # tests
        with self.assertRaises(Exception):    
            _ = Polynomial(xtrain.reshape(-1,1), ytrain, 10)

        with self.assertRaises(Exception):    
            _ = Polynomial(xtrain, ytrain.reshape(-1,1), 10)

        _ = Polynomial(xtrain.reshape(-1,1), ytrain.reshape(-1,1), 10)

        with self.assertRaises(Exception):
            _ = Polynomial(xtrain.reshape(-1,1), ytrain.reshape(-1,1), -1)

if __name__ == '__main__':
    unittest.main()

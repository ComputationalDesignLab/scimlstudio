import unittest, torch
from scimlstudio.models import FeedForwardNeuralNetwork
from scimlstudio.utils import evaluate_scalar, Standardize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
args = {
    "device": device,
    "dtype": dtype
}

class TestFeedForwardNeuralNetwork(unittest.TestCase):
    """
        Class defining the test cases for the feed forward neural network model
    """

    def test_nn_model_1d(self):

        # training data
        xtrain = torch.linspace(0, 2*torch.pi, 7, **args).reshape(-1,1)
        ytrain = torch.sin(xtrain)

        # testing data
        xtest = torch.linspace(0, 2*torch.pi, 100, **args).reshape(-1,1)
        ytest = torch.sin(xtest)

        # network
        network = torch.nn.Sequential(
            torch.nn.Linear(in_features=xtrain.shape[1], out_features=32),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=32, out_features=32),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=32, out_features=32),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=32, out_features=ytrain.shape[1]),
        ).to(**args)

        def init_weights(m):
            """
                Function for initializing the weights using glorot (or xavier) initialization
            """

            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.0)

        # initial weights
        network.apply(init_weights)

        # data transforms
        input_transform = Standardize(xtrain)
        output_transform = Standardize(ytrain)

        # create model instance
        model = FeedForwardNeuralNetwork(xtrain, ytrain, network, input_transform=input_transform, output_transform=output_transform)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters, lr=0.01)

        # loss function
        loss_func = torch.nn.MSELoss()

        # fit the model
        model.fit(optimizer, loss_func, batch_size=xtrain.shape[0], epochs=100)

        # predict
        ytest_pred = model.predict(xtest)

        # metrics
        r2 = evaluate_scalar(ytest.reshape(-1,), ytest_pred.reshape(-1,), "r2")
        nrmse = evaluate_scalar(ytest.reshape(-1,), ytest_pred.reshape(-1,), "nrmse")

        assert nrmse < 2e-2 and r2 > 0.99

    def test_nn_model_2d(self):

        # train
        x1 = torch.linspace(0,1,5,**args)
        x2 = torch.linspace(0,1,5,**args)
        X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
        xtrain = torch.hstack(( X1.reshape(-1,1), X2.reshape(-1,1) ))
        ytrain = torch.cos(torch.sum(xtrain, axis=1))*torch.exp(torch.prod(xtrain, axis=1))
        ytrain = ytrain.reshape(-1,1)

        # test
        x1 = torch.linspace(0,1,15,**args)
        x2 = torch.linspace(0,1,15,**args)
        X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
        xtest = torch.hstack(( X1.reshape(-1,1), X2.reshape(-1,1) ))
        ytest = torch.cos(xtest[:,0]+xtest[:,1])*torch.exp(xtest[:,0]*xtest[:,1])
        ytest = ytest.reshape(-1,1)

        # network
        network = torch.nn.Sequential(
            torch.nn.Linear(in_features=xtrain.shape[1], out_features=32),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=32, out_features=32),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=32, out_features=32),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=32, out_features=ytrain.shape[1]),
        ).to(**args)

        def init_weights(m):
            """
                Function for initializing the weights using glorot (or xavier) initialization
            """

            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.0)

        # initial weights
        network.apply(init_weights)

        # data transforms
        input_transform = Standardize(xtrain)
        output_transform = Standardize(ytrain)

        # create model instance
        model = FeedForwardNeuralNetwork(xtrain, ytrain, network, input_transform=input_transform, output_transform=output_transform)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters, lr=0.01)

        # loss function
        loss_func = torch.nn.MSELoss()

        # fit the model
        model.fit(optimizer, loss_func, batch_size=xtrain.shape[0], epochs=100)

        # predict
        ytest_pred = model.predict(xtest)

        # metrics
        r2 = evaluate_scalar(ytest.reshape(-1,), ytest_pred.reshape(-1,), "r2")
        nrmse = evaluate_scalar(ytest.reshape(-1,), ytest_pred.reshape(-1,), "nrmse")

        assert nrmse < 1e-2 and r2 > 0.99

    def test_input_output_shapes(self):

        # dummy training data
        xtrain = torch.rand(10, 5, **args)
        ytrain = torch.rand(10, 1, **args)

        # network
        network = torch.nn.Sequential(
            torch.nn.Linear(in_features=xtrain.shape[1], out_features=16),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=16, out_features=16),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=16, out_features=ytrain.shape[1]),
        ).to(**args)

        # create model instance
        model = FeedForwardNeuralNetwork(xtrain, ytrain, network)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters, lr=0.01)

        # loss function
        loss_func = torch.nn.MSELoss()

        # fit the model
        model.fit(optimizer, loss_func, batch_size=xtrain.shape[0], epochs=100)

        # predict - 1 samples
        ypred = model.predict(xtrain[0])
        assert ypred.ndim == 1 and ypred.shape[0] == 1

        # predict - 5 samples
        ypred = model.predict(xtrain[:5])
        assert ypred.ndim == 2 and ypred.shape[0] == 5

if __name__ == '__main__':
    unittest.main()

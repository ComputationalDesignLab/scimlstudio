import unittest, torch
from scimlstudio.models import MultifidelityNeuralNetwork
from scimlstudio.utils import evaluate_scalar, Standardize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
args = {
    "device": device,
    "dtype": dtype
}

def low_fidelity(x):
    return 0.5*(6*x - 2)**2 * torch.sin(12*x - 4) + 10*(x - 0.5) - 5

def high_fidelity(x):
    return (6*x - 2)**2 * torch.sin(12*x - 4)

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)

x_lf = torch.linspace(0, 1, 11, **args).reshape(-1,1)
y_lf = low_fidelity(x_lf)

x_hf = torch.tensor([0.0, 0.4, 0.6, 1.0], **args).reshape(-1,1)
y_hf = high_fidelity(x_hf)

x_test = torch.linspace(0, 1, 100, **args).reshape(-1,1)
y_test_hf = high_fidelity(x_test)

# transforms
input_transform = Standardize(torch.vstack((x_lf,x_hf)))
output_transform_lf = Standardize(y_lf)
output_transform_hf = Standardize(y_hf)

# mf networks
network_lf = torch.nn.Sequential(
    torch.nn.Linear(x_lf.shape[1], 128),
    torch.nn.Tanh(),
    torch.nn.Linear(128, 128),
    torch.nn.Tanh(),
    torch.nn.Linear(128, y_lf.shape[1])
).to(**args)

network_lf.apply(weights_init)

network_linear_corr = torch.nn.Sequential(
    torch.nn.Linear(x_hf.shape[1] + y_lf.shape[1], 64),
    torch.nn.Linear(64, 64),
    torch.nn.Linear(64, y_hf.shape[1])
).to(**args)

network_linear_corr.apply(weights_init)

network_nonlinear_corr = torch.nn.Sequential(
    torch.nn.Linear(x_hf.shape[1] + y_lf.shape[1], 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, y_hf.shape[1])
).to(**args)

network_nonlinear_corr.apply(weights_init)

class TestMultiFidelityNeuralNetwork(unittest.TestCase):
    """
        Class defining test cases for the multi-fidelity neural network model
    """

    def test_mfnn_model(self):

        # initialize mf model
        mf_model = MultifidelityNeuralNetwork(
            x_train_lf=x_lf,
            y_train_lf=y_lf,
            x_train_hf=x_hf,
            y_train_hf=y_hf,
            network_lf=network_lf,
            network_linear_corr=network_linear_corr,
            network_nonlinear_corr=network_nonlinear_corr,
            input_transform=input_transform,
            output_transform_lf=output_transform_lf,
            output_transform_hf=output_transform_hf
        )

        optimizer = torch.optim.Adam(mf_model.parameters, lr=1e-3)

        mf_model.fit(
            optimizer=optimizer,
            epochs=5000,
            reg_const_lf=1e-4,
            reg_const_nonlinear=1e-3
        )

        ytest_pred = mf_model.predict(x_test.reshape(-1,1))

        # metrics
        r2 = evaluate_scalar(y_test_hf.reshape(-1,), ytest_pred.reshape(-1,), "r2")
        nrmse = evaluate_scalar(y_test_hf.reshape(-1,), ytest_pred.reshape(-1,), "nrmse")

        print(r2, nrmse)

        assert nrmse < 2.5e-2 and r2 > 0.99

if __name__ == '__main__':
    unittest.main()
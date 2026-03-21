import unittest, torch
from scimlstudio.models import FeedForwardAutoencoder
from scimlstudio.utils import Standardize, evaluate_vector

args = {"device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'), "dtype": torch.float32}

class TestFeedForwardAutoencoder(unittest.TestCase):
    """
        Class defining the test cases for the FeedForwardAutoencoder model
    """

    def test_reconstruction(self):

        # generating random training data
        xtrain = torch.randn(100,1000)

        # function to initialize the weights using glorot (or xavier) initialization
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight) # change this for other init methods
                m.bias.data.fill_(0.0)

        # defining the encoder and decoder networks
        encoder = torch.nn.Sequential(
            torch.nn.Linear(xtrain.shape[1], 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 16)
        ).to(**args)

        decoder = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, xtrain.shape[1])
        ).to(**args)

        # initializing the networks
        encoder.apply(init_weights)
        decoder.apply(init_weights)

        # standardizing the data
        data_transform = Standardize(xtrain)

        autoencoder = FeedForwardAutoencoder(x_train=xtrain, encoder=encoder, decoder=decoder, data_transform=data_transform)

        # training the autoencoder
        optimizer = torch.optim.Adam(autoencoder.parameters, lr=1e-3)
        loss_func = torch.nn.MSELoss() 
        autoencoder.fit(optimizer=optimizer, loss_func=loss_func, batch_size=xtrain.shape[0], epochs=1000)
        
        # generate encoding
        z = autoencoder.encoding(xtrain)
        assert z.shape[-1] < xtrain.shape[-1], "the encoder must produce latent variables of lower dimension than the original data"

        # generate decoding
        reconstruction = autoencoder.decoding(z)
        assert reconstruction.shape == xtrain.shape, "the decoder output must have the same shape as the original data"

        # test prediction shape
        predict = autoencoder.predict(xtrain)
        assert predict.shape[-1] == xtrain.shape[-1], "autoencoder reconstruction must have the same number of outputs as the original data" 

    def test_input_output_shapes(self):

        # generating random training data
        xtrain = torch.randn(100,1000)
        transform = Standardize(xtrain)

        # defining the encoder and decoder networks
        encoder = torch.nn.Sequential(
            torch.nn.Linear(xtrain.shape[1], 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 16)
        ).to(**args)

        decoder = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, xtrain.shape[1])
        ).to(**args)

        autoencoder = FeedForwardAutoencoder(x_train=xtrain, encoder=encoder, decoder=decoder, data_transform=transform)

        # training the autoencoder
        optimizer = torch.optim.Adam(autoencoder.parameters, lr=1e-3)
        loss_func = torch.nn.MSELoss() 
        autoencoder.fit(optimizer=optimizer, loss_func=loss_func, batch_size=xtrain.shape[0], epochs=1000)

        # prediction for one sample
        xpred = autoencoder.predict(xtrain[0])
        assert xpred.ndim == 1 and xpred.shape[0] == xtrain.shape[-1]

        # prediction for 50 samples
        xpred = autoencoder.predict(xtrain[:50])
        assert xpred.ndim == 2 and xpred.shape[0] == 50

    def test_inputs(self):

        # Generate dummy data
        x_random = torch.rand(200,2000)
        transform = Standardize(x_random)

         # defining the encoder and decoder networks
        encoder = torch.nn.Sequential(
            torch.nn.Linear(x_random.shape[1], 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 16)
        ).to(**args)

        decoder = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, x_random.shape[1])
        ).to(**args)

        with self.assertRaises(Exception):
            _ = FeedForwardAutoencoder(x_random.reshape(-1,), encoder=encoder, decoder=decoder, data_transform=None)

        with self.assertRaises(Exception):
            _ = FeedForwardAutoencoder(x_random, encoder=None, decoder=None, data_transform=transform)

        with self.assertRaises(Exception):
            _ = FeedForwardAutoencoder(x_random, encoder=encoder, decoder=decoder, data_transform=decoder)

        with self.assertRaises(Exception):
            _ = FeedForwardAutoencoder(x_random, encoder=encoder, decoder=decoder, data_transform=1.0)

        _ = FeedForwardAutoencoder(x_random, encoder=encoder, decoder=decoder, data_transform=transform)

if __name__ == '__main__':
    unittest.main()



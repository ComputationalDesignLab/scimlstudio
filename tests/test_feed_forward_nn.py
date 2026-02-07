import unittest, torch
from scimlstudio.models import FeedForwardNeuralNetwork
from scimlstudio.utils import evaluate_scalar, Standardize, Normalize
from pyDOE3 import lhs, halton_sequence

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

    def test_nn_model(self):

        pass

    def test_input_output_shapes(self):

        pass

if __name__ == '__main__':
    unittest.main()

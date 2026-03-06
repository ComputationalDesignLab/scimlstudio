import unittest, torch
from scimlstudio.models import POD
from scimlstudio.utils import Standardize

args = {"device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'), "dtype": torch.float32}

class TestPOD(unittest.TestCase):
    """
        Class defining the test cases for the POD model
    """

    def test_projection(self):

        # generating random training data
        xtrain = torch.randn(100,1000)

        # creating the POD model
        pod = POD(s_train=xtrain.mT, ric=0.95, snapshot_transform=None)
        pod.fit()
        assert pod.k < xtrain.shape[0], "size of POD projection must be less than number of data samples"
        
        # generate encoding
        z = pod.encoding(xtrain.mT)
        assert z.shape[-1] < xtrain.shape[-1], "the projection step must lead to a lower dimension than the original data"

        # check ric
        assert torch.all(torch.tensor(pod.ric_list, **args)) <= 1.0, "ric values must be always less than 1.0"

        # check POD modes
        assert pod.modes.shape[0] == xtrain.shape[-1], "pod modes must have the same number of outputs as the original data" 

        # test prediction shape
        predict = pod.predict(xtrain.mT)
        assert predict.shape[0] == xtrain.shape[-1], "pod reconstruction must have the same number of outputs as the original data" 

    def test_projection_with_transform(self):

        # generating random training data
        xtrain = torch.randn(100,1000)
        transform = Standardize(xtrain)

        # creating the POD model
        pod = POD(s_train=xtrain.mT, ric=0.9999, snapshot_transform=transform)
        pod.fit()
        assert pod.k < xtrain.shape[0], "size of POD projection must be less than number of data samples"
        
        # generate encoding
        z = pod.encoding(xtrain.mT)
        assert z.shape[-1] < xtrain.shape[-1], "the projection step must lead to a lower dimension than the original data"

        # check ric
        assert torch.all(torch.tensor(pod.ric_list, **args)) <= 1.0, "ric values must be always less than 1.0"

        # check POD modes
        assert pod.modes.shape[0] == xtrain.shape[-1], "pod modes must have the same number of outputs as the original data" 

        # test prediction shape
        predict = pod.predict(xtrain.mT)
        assert predict.shape[0] == xtrain.shape[-1], "pod reconstruction must have the same number of outputs as the original data" 

    def test_inputs(self):

        # Generate dummy data
        x_random = torch.rand(200,2000)
        transform = Standardize(x_random)

        with self.assertRaises(Exception):
            _ = POD(x_random.reshape(-1,), ric = 0.95, snapshot_transform=None)

        with self.assertRaises(Exception):
            _ = POD(x_random, ric = -0.95, snapshot_transform=None)

        with self.assertRaises(Exception):
            _ = POD(x_random.mT, ric = 3.0, snapshot_transform=None)

        with self.assertRaises(Exception):
            _ = POD(x_random.mT, ric = 0.95, snapshot_transform=1.0)

        _ = POD(x_random.mT, ric = 0.95, snapshot_transform=transform)

if __name__ == '__main__':
    unittest.main()



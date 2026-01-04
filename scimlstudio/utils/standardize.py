import torch

class Standardize:

    def __init__(self, x: torch.Tensor, mean: torch.Tensor = None, std: torch.Tensor = None):

        """
            Class for standardization of data

            Parameters
            ----------
            x : torch.Tensor
                data that is to be standardized

            mean : torch.Tensor
                1D tensor which contains the mean value of each column in x. If mean = None, then it is 
                computed within this function.

            std : torch.Tensor
                1D tensor which contains the standard deviation of each column in x. If std = None, then it is 
                computed within this function.

        """

        # Checks for the inputs
        assert isinstance(x, torch.Tensor), "Input data must be a torch Tensor"
        assert x.ndim == 2, "Input data must be a 2D tensor"
        assert isinstance(mean, torch.Tensor) or isinstance(mean, type(None)), "mean should be a torch tensor or None"
        assert isinstance(std, torch.Tensor) or isinstance(std, type(None)), "std should be a torch tensor or None"

        # Checking the inputs and computing them if required
        if mean is None:
            self.mean = torch.mean(x, dim=0)
        else:
            assert mean.ndim == 1, "mean must be a 1D tensor"
            assert mean.shape[0] == x.shape[1], "mean must have the same number of features as input data"
            self.mean = mean

        if std is None:
            # Clamping standard deviation to a very small value to avoid a divide by zero
            self.std = torch.clamp(torch.std(x, dim=0), 1e-8)
        else:
            assert std.ndim == 1, "std must be a 1D tensor"
            assert std.shape[0] == x.shape[1], "std must have the same number of features as input data"
            assert torch.all(std > 0.0), "std must be clamped to a very small value to avoid a divide by zero error"
            self.std = std

        assert x.device == self.mean.device == self.std.device, "Input data, mean and std must be on the same device"
        assert self.mean.shape == self.std.shape, "mean and std must have the same shape"

    def transform(self, x: torch.Tensor) -> torch.Tensor:

        """
            Standardize the data

            Parameters
            ----------
            x : torch.Tensor
                data to be standardized

            Returns
            -------
            x : torch.Tensor
                standardized data
        """
        assert isinstance(x, torch.Tensor), "Input data must be a torch Tensor"
        assert x.ndim == 2, "Input data must be a 2D tensor"
        assert x.shape[1] == self.mean.shape[0], "Input data must have the same number of features as the data used for fitting"
        assert x.device == self.mean.device == self.std.device, "Input data, mean and std must be on the same device"

        return (x - self.mean) / self.std
        
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:

        """
            Inverse the standardization to obtain original data

            Parameters
            ----------
            x : torch.Tensor
                standardized data that needs to be inverse transformed

            Returns
            -------
            x : torch.Tensor
                data with standardization removed
        """
        assert isinstance(x, torch.Tensor), "Input data must be a torch Tensor"
        assert x.ndim == 2, "Input data must be a 2D tensor"
        assert x.shape[1] == self.mean.shape[0], "Input data must have the same number of features as the data used for fitting"
        assert x.device == self.mean.device == self.std.device, "Input data, mean and std must be on the same device"

        return x * self.std + self.mean

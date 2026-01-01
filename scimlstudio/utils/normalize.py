import torch

class Normalize:

    def __init__(self, x: torch.Tensor, min: torch.Tensor = None, max: torch.Tensor = None):
        """
            Class for Min-max normalization of data

            Parameters
            ----------
            x : torch.Tensor
                data to be normalized

            min: torch.Tensor
                1D tensor array which defines lower bound for normalization. If none, then
                min is set to 0 for each feature

            max: torch.Tensor
                1D tensor array which defines upper bound for normalization. If none, then
                max is set to 1 for each feature
        """

        # Some checks
        assert isinstance(x, torch.Tensor), "Input data must be a torch Tensor"
        assert x.ndim == 2, "Input data must be a 2D tensor"
        assert isinstance(min, type(None)) or isinstance(min, torch.Tensor), "min must be a torch Tensor or None"
        assert isinstance(max, type(None)) or isinstance(max, torch.Tensor), "max must be a torch Tensor or None"

        self.min_feature_vector = torch.min(x, axis=0)[0] # min value of each feature
        self.max_feature_vector = torch.max(x, axis=0)[0] # max value of each feature

        if min is None:
            self.min = torch.zeros(x.shape[1]).to(x)
        else:
            assert min.ndim == 1, "min must be a 1D tensor"
            assert min.shape[0] == x.shape[1], "min must have the same number of features as the input data"
            self.min = min

        if max is None:
            self.max = torch.ones(x.shape[1]).to(x)
        else:
            assert max.ndim == 1, "max must be a 1D tensor"
            assert max.shape[0] == x.shape[1], "max must have the same number of features as the input data"
            self.max = max

        assert torch.all(self.max > self.min), "All elements of max must be greater than min"

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
            Transform the data to the given range [min, max]

            Parameters
            ----------
            x : torch.Tensor
                data to be transformed

            Returns
            -------
            x : torch.Tensor
                data scaled to the given range [min, max]
        """

        assert isinstance(x, torch.Tensor), "Input data must be a torch Tensor"
        assert x.ndim == 2, "Input data must be a 2D tensor"
        assert x.shape[1] == self.min_feature_vector.shape[0], "Input data must have the same number of features as the data used for fitting"

        x = (x - self.min_feature_vector) / (self.max_feature_vector - self.min_feature_vector) # scaling to [0, 1]

        x = x * (self.max - self.min) + self.min # scaling to [min, max]

        return x

    def inverse_transform(self, x:torch.Tensor) -> torch.Tensor:
        """
            Inverse transform the data to the original range

            Parameters
            ----------
            x : torch.Tensor
                scaled data to be inverse tranformed

            Returns
            -------
            x : torch.Tensor
                data scaled to original range
        """

        assert isinstance(x, torch.Tensor), "Input data must be a torch Tensor"
        assert x.ndim == 2, "Input data must be a 2D tensor"
        assert x.shape[1] == self.min_feature_vector.shape[0], "Input data must have the same number of features as the data used for fitting"

        x = (x - self.min) / (self.max - self.min) # scaling to [0, 1]

        x = x * (self.max_feature_vector - self.min_feature_vector) + self.min_feature_vector # scaling to original range

        return x

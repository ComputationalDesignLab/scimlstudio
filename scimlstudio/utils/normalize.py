import torch

class Normalize:

    def __init__(self, x: torch.Tensor, lower_bound: torch.Tensor = None, upper_bound: torch.Tensor = None):
        """
            Class for Min-max normalization of data

            Parameters
            ----------
            x : torch.Tensor
                data to be normalized

            lower_bound: torch.Tensor
                1D tensor array which defines lower bound for normalization. If none, then
                lower_bound is set to 0 for each feature

            upper_bound: torch.Tensor
                1D tensor array which defines upper bound for normalization. If none, then
                upper_bound is set to 1 for each feature
        """

        # Some checks
        assert isinstance(x, torch.Tensor), "Input data must be a torch Tensor"
        assert x.ndim == 2, "Input data must be a 2D tensor"
        assert isinstance(lower_bound, type(None)) or isinstance(lower_bound, torch.Tensor), "lower_bound must be a torch Tensor or None"
        assert isinstance(upper_bound, type(None)) or isinstance(upper_bound, torch.Tensor), "upper_bound must be a torch Tensor or None"

        self.min_feature_vector = torch.min(x, axis=0)[0] # min value of each feature
        self.max_feature_vector = torch.max(x, axis=0)[0] # max value of each feature

        if lower_bound is None:
            self.lower_bound = torch.zeros(x.shape[1]).to(x)
        else:
            assert lower_bound.ndim == 1, "lower_bound must be a 1D tensor"
            assert lower_bound.shape[0] == x.shape[1], "lower_bound must have the same number of features as the input data"
            self.lower_bound = lower_bound.to(x)

        if upper_bound is None:
            self.upper_bound = torch.ones(x.shape[1]).to(x)
        else:
            assert upper_bound.ndim == 1, "upper_bound must be a 1D tensor"
            assert upper_bound.shape[0] == x.shape[1], "upper_bound must have the same number of features as the input data"
            self.upper_bound = upper_bound.to(x)

        assert torch.all(self.upper_bound > self.lower_bound), "All elements of upper_bound must be greater than lower_bound"

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
            Transform the data to the given range [lower_bound, upper_bound]

            Parameters
            ----------
            x : torch.Tensor
                data to be transformed

            Returns
            -------
            x : torch.Tensor
                data scaled to the given range [lower_bound, upper_bound]
        """

        assert isinstance(x, torch.Tensor), "Input data must be a torch Tensor"
        assert x.ndim == 2, "Input data must be a 2D tensor"
        assert x.shape[1] == self.min_feature_vector.shape[0], "Input data must have the same number of features as the data used for fitting"

        x = (x - self.min_feature_vector) / (self.max_feature_vector - self.min_feature_vector) # scaling to [0, 1]

        x = x * (self.upper_bound - self.lower_bound) + self.lower_bound # scaling to [lower bound, upper bound]

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

        x = (x - self.lower_bound) / (self.upper_bound - self.lower_bound) # scaling to [0, 1]

        x = x * (self.max_feature_vector - self.min_feature_vector) + self.min_feature_vector # scaling to original range

        return x

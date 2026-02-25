import torch
from ..base_models import BaseDimensionalityReduction
from ..utils import Standardize, Normalize

class POD(BaseDimensionalityReduction):

    def __init__( 
            self,
            s_train: torch.Tensor,
            ric: float,
            snapshot_transform: Standardize | Normalize | None = None
    ):
        """
            Definition of the class for proper orthogonal decomposition using 
            singular value decomposition. This is a dimensionality reduction 
            and unsupervised learning method.

            Parameters
            ----------
            s_train: torch.Tensor
                Training snapshots for the POD method
                This must be of shape (D x N) where D is the dimensionality of each snapshot
                and N is the number of snapshots
            ric: flot
                Value of the relative information content
            snapshot_transform: Standardize or Normalize or None
                Data transform that must be applied to the snapshots
        """

        # Setting the snapshots
        assert isinstance(s_train, torch.Tensor) and s_train.dim == 2, "Snapshots must be provided as a 2D tensor array"
        self.s_train = s_train

        # Setting the ric value
        assert isinstance(ric, float) and ric >= 0.0 and ric <= 1.0, "RIC must be a floating point value between 0 and 1"
        self.ric = ric

        # Setting the snapshot transform
        if snapshot_transform is not None:
            assert isinstance(snapshot_transform, Normalize) or isinstance(snapshot_transform, Standardize), " Snapshot transform must be an instance of Normalize or Standardize class"
        self.snapshot_transform = snapshot_transform

    def calculate_truncation(self, sigma: torch.Tensor):
        """
            Class method to calculate the number of modes to truncate to
            for maintaining the specified RIC value

            Parameters
            ----------
            sigma: torch.Tensor
                Singular values in 1D tensor array

            Returns
            -------
            k: int
                Number of POD modes for truncation
        """
        assert isinstance(sigma,torch.Tensor), "the provided singular values must be a tensor array"

        # Squared singular values
        sigma_squared = torch.square(sigma)

        for k in range(len(sigma)):
            # Calculating the ric
            calculated_ric = torch.sum(torch.square(sigma[:k])) / torch.sum(sigma_squared)
            # Comparing the ric
            if calculated_ric < self.ric:
                break
            else:
                continue

        return k

    def fit(self, full_svd: bool = True):
        """
            Class method to calculate the POD modes and coefficients
            Essentially, fitting the POD to the snapshot data

            Parameters
            ----------
            full_svd: bool
                Boolean value that specifies whether to perform full or reduced SVD
        """
        # Performing singular value decomposition
        snapshots = self.snapshot_transform.transform(self.s_train) if self.snapshot_transform is not None else self.s_train
        self.U, self.S, self.Vt = torch.linalg.svd(snapshots, full_matrices = full_svd)

        # Calculating the truncation modes based on RIC
        self.k = self.calculate_truncation(self.S)
        # Truncating the modes
        self.modes = self.U[:self.k]

    def encoding(self, x: torch.Tensor):
        """
            Class method to encode data using the developed POD modes
            In terms of POD, the encoding is calculating the low dimensional
            coordinates

            Parameters
            ----------
            x: torch.Tensor
                2D Tensor array containing the data to be encoded

            Returns
            -------
            coord: torch.Tensor
                2D Tensor array containing the coordinates for the data provided
        """
        assert isinstance(x, torch.Tensor) and x.ndim==2, "x must be 2D Tensor array"
        assert x.device == self.s_train.device, "the given data must be on the same device as the training snapshots"
        # Transforming the input
        if self.snapshot_transform is not None:
            x = self.snapshot_transform.transform(x)
        # Calculating the coordinates
        coord = torch.matmul(self.U, x)
        return coord
    
    def decoding(self, coord: torch.Tensor):
        """
            Class method to decode coefficients to recover the original 
            high-dimensional data

            Parameters
            ----------
            coord: torch.Tensor
                2D Tensor array containing the coordinates

            Returns
            -------
            y : torch.Tensor
                2D Tensor array containing reconstructions of the high dimensional data
        """
        assert isinstance(coord, torch.Tensor) and coord.ndim==2, "corodinates must be 2D Tensor array"
        assert coord.device == self.s_train.device, "the given data must be on the same device as the training snapshots"
        # Calculating the reconstruction
        y = torch.matmul(self.U, coord.mT)
        # Transforming the reconstruction
        if self.snapshot_transform is not None:
            y = self.snapshot_transform.inverse_transform(y)
        return y
    
    def predict(self, x: torch.Tensor):
        """
            Class method for prediction using the POD method

            Parameters
            ----------
            x: torch.Tensor
                2D Tensor array containing the high-dimensional data

            Returns
            -------
            reconstruction: torch.Tensor
                2D Tensor array containing the reconstructions of the data
        """
        assert isinstance(x, torch.Tensor) and x.ndim==2, "x must be 2D Tensor array"
        assert x.device == self.s_train.device, "the given data must be on the same device as the training snapshots"

        with torch.no_grad():
            if self.snapshot_transform is not None:
                x = self.snapshot_transform.transform(x)
            coord = torch.matmul(self.U, x)
            reconstruction = torch.matmul(self.U, coord.mT)
            if self.snapshot_transform is not None:
                reconstruction = self.snapshot_transform.inverse_transform(reconstruction)
            return reconstruction

        





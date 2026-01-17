import torch
from ..base_model import BaseModel
from ..utils.transformations import Standardize, Normalize

class RBF(BaseModel):

    def __init__(self, xtrain: torch.Tensor, ytrain: torch.Tensor, sigma: float,
                 input_transform: Normalize | Standardize | None = None, output_transform: Normalize | Standardize | None = None, 
                 basis: str = "gaussian"):

        """
            Class definition for parametric radial basis function models

            f(x) = w^T psi(x), psi_i(x) = psi(r), where r = ||x - c_i||

            NOTE: Only the Gaussian and multiquadric basis functions are implemented

            Parameters
            ----------
            xtrain: torch.Tensor
                Input training data for the RBF model with size (N, d)

            ytrain: torch.Tensor
                Output training data for the RBF model with size (N, 1)

            sigma: float
                Value of sigma for calculating the basis function

            input_transform:  Normalize | Standardize | None
                Object of a transformation class for applying a transform to the inputs of the model
                The default is None which means the application of no transform to inputs

            output_transform:  Normalize | Standardize | None
                Object of a transformation for applying a transform to the outputs of the model
                The default is None which means the application of no transform to outputs

            basis: str
                String that specifies which parametric basis function should be used
                User has a choice between "gaussian", "multiquadric" and "inverse"

                gaussian --> exp(- r^2/(2 * sigma^2))
                multiquadric --> (r^2 + sigma^2)^0.5
                inverse --> (r^2 + sigma^2)^(-0.5)        
        """

        super().__init__()
        self.sigma = sigma
        self.basis = basis
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.basis_function = self.set_basis_function(self.basis)
        
        # Set training data for the model
        self.xtrain = self.transform_values(xtrain, input_transform)
        self.ytrain = self.transform_values(ytrain, output_transform)

    def fit(self) -> torch.Tensor:

        """
            Method for fitting the model to the provided training data

            Returns
            -------
            rbf_weights: torch.Tensor
                Tensor array with the fitted weights of the model
        """
        
        # Basis matrix definition
        ns = self.xtrain.shape[0]
        basis_matrix = torch.zeros((ns, ns)).to(self.xtrain)

        # Assigning values to the basis matrix
        for i in range(ns):
            for j in range(ns):
                # Calculating the value of r
                r = torch.linalg.norm(self.xtrain[i] - self.xtrain[j], ord=2)

                # Putting value in basis matrix 
                basis_matrix[i, j] = self.basis_function(r)
                
        basis_matrix_inverse = torch.linalg.pinv(basis_matrix)
        self.rbf_weights = torch.matmul(basis_matrix_inverse, self.ytrain)

        return self.rbf_weights.clone()
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:

        """
            Method for to predict values from the model for specified points

            Parameters
            ----------
            x: torch.Tensor
                Data for which predictions must be made. Must be a 2D tensor array

            Returns
            -------
            ypred: torch.Tensor
                Predictions of the model on the specified points
        """
        x = self.transform_values(x, self.input_transform)

        # Basis matrix definition
        basis_matrix = torch.zeros((self.xtrain.shape[0], x.shape[0])).to(self.xtrain)

        # Assigning values to the basis matrix
        for i in range(self.xtrain.shape[0]):
            for j in range(x.shape[0]):
                r = torch.linalg.norm(x[i] - self.xtrain[j], ord=2)
                basis_matrix[i, j] = self.basis_function(r)

        ypred = torch.matmul(basis_matrix, self.rbf_weights)
        ypred = self.transform_values(ypred, self.output_transform)

        return ypred
        
    def set_basis_function(self, basis: str): 

        """
            Method for setting the basis function of the RBF model based on user input

            Parameters
            ----------
            basis: str
                String that specifies the user choice for the basis function

            Returns
            -------
            basis_function: 
                Python function that calculates the values of the basis function based on the value of r
        """
        if basis == "gaussian":
            basis_function = lambda r: torch.exp(-r**2/(2 * self.sigma ** 2))
        elif basis == "multiquadric":
            basis_function = lambda r: torch.sqrt(r**2 + self.sigma ** 2)
        elif basis == "inverse":
            basis_function = lambda r: 1 / torch.sqrt(r**2 + self.sigma ** 2)

        return basis_function

    def transform_values(self, data: torch.Tensor, transform: Normalize | Standardize | None) -> torch.Tensor:

        """
            Method for transforming values based on given transform

            Parameters
            ----------
            data: torch.Tensor
                Data that must be transformed

            transform: Normalize | Standardize | None
                Object of a transformation class that will be used to transform the data

            Returns
            -------
            transformed_data: torch.Tensor
                Data after the application of the transform
                If transform is None, then the original data is returned
        """

        # Checking if transform is None and applying the transform accordingly
        if transform is None:
            transformed_data = data
        else:
            transformed_data = transform.transform(data)

        return transformed_data

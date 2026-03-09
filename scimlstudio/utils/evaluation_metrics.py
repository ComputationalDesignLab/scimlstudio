import torch

def evaluate_scalar(true_values: torch.Tensor, predict_values: torch.Tensor, metric: str = "rmse") -> float:
    """
        Function to evaluate difference between predicted and true values

        NOTE: this function is only for scalar prediction

        Parameters
        ----------
        true_values: torch.Tensor
            1D tensor array containing true values

        predict_values: torch.Tensor
            1D tensor array containing predicted values
        
        metric: str
            method to use for comparing true and predicted values. Valid
            metrics are "rmse", "nrmse", and "r2"

        Returns
        -------
        z: float
            computed metric for given true and predicted values
    """

    assert isinstance(true_values, torch.Tensor) and true_values.ndim == 1, "'true values' should be a 1D tensor array"
    assert true_values.shape[0] == predict_values.shape[0], "true and predicted values should be of same size"
    assert isinstance(predict_values, torch.Tensor) and predict_values.ndim == 1, "'predict values' should be a 1D tensor array"
    assert true_values.device == predict_values.device, "both true and predict values should be on same device"
    assert isinstance(metric, str) and metric.lower() in ["rmse", "nrmse", "r2"], "metric should be from 'rmse', 'nrmse', and 'r2'"

    if metric == "nrmse" or metric == "rmse":

        mse = torch.sum((true_values - predict_values)**2) / true_values.shape[0]

        rmse = mse**0.5

        if metric == "rmse":
            return rmse.item()
        
        if metric == "nrmse":
            nrmse = rmse / (torch.max(true_values) - torch.min(true_values))
            return nrmse.item()
        
    if metric == "r2":

        covariance = ( (true_values - torch.mean(true_values)) * 
                      (predict_values - torch.mean(predict_values)) ).sum() / (true_values.shape[0] - 1)

        prod_var = torch.var(true_values) * torch.var(predict_values)

        r2 = covariance**2 / prod_var

        return r2.item()
    
def evaluate_vector(true_values: torch.Tensor, predict_values: torch.Tensor, metric: str = "rmse") -> torch.Tensor:
    """
        Function to evaluate difference between predicted and true values

        NOTE: this function is only for vector prediction

        Parameters
        ----------
        true_values: torch.Tensor
            2D tensor array containing true values.
            the dimensions must be n x o where n is number of testing samples
            and o is the number of outputs in the vector output. 

        predict_values: torch.Tensor
            2D tensor array containing predicted values.
            the dimensions must be n x o where n is number of testing samples
            and o is the number of outputs in the vector output. 
        
        metric: str
            method to use for comparing true and predicted values. Valid
            metrics are "rmse" and "nrmse"

        Returns
        -------
        z: torch.Tensor
            computed metric for given true and predicted values over the entire grid
    """

    assert isinstance(true_values, torch.Tensor) and true_values.ndim == 2, "'true values' should be a 2D tensor array"
    assert true_values.shape[0] == predict_values.shape[0], "true and predicted values should have the same number of samples"
    assert isinstance(predict_values, torch.Tensor) and predict_values.ndim == 2, "'predict values' should be a 2D tensor array"
    assert true_values.device == predict_values.device, "both true and predict values should be on same device"
    assert isinstance(metric, str) and metric.lower() in ["rmse", "nrmse"], "metric should be from 'rmse' and 'nrmse'"

    if metric == "nrmse" or metric == "rmse":

        rmse = torch.sqrt(torch.mean((true_values - predict_values)**2, dim=0))**0.5
        if metric == "rmse":
            return rmse
        
        if metric == "nrmse":
            max_true_vals, _ = torch.max(true_values, dim=0)
            min_true_vals, _ = torch.min(true_values, dim=0)
            nrmse = rmse / (torch.max(max_true_vals) - torch.min(min_true_vals))
            return nrmse
        


        
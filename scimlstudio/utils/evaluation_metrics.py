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
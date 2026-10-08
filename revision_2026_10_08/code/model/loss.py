"""One metric implementation for training, selection, and reporting.

The default includes physically valid zero and negative targets. Historical
zero exclusion is available ONLY through an explicit null_val=0 argument.
Nonfinite predictions on valid targets raise instead of silently disappearing.
MAPE is auxiliary and excludes |target| <= eps; it is not a primary paper metric.
"""
import torch


def _valid(y_pred, y_true, null_val=None):
    if y_pred.shape != y_true.shape:
        raise ValueError('Prediction and target shapes must match.')
    valid = torch.isfinite(y_true)
    if null_val is not None:
        valid = valid & (y_true != null_val)
    if torch.any(valid & ~torch.isfinite(y_pred)):
        raise ValueError('Nonfinite prediction at a valid target.')
    if not torch.any(valid):
        raise ValueError('No valid targets remain under the requested metric protocol.')
    return valid


def _reduce(y_pred, y_true, power, horizon_weights=None, null_val=None):
    valid = _valid(y_pred, y_true, null_val)
    # Select safe values before subtraction so masked NaNs cannot poison gradients.
    diff = torch.where(valid, y_pred, 0.0) - torch.where(valid, y_true, 0.0)
    loss = diff.abs() if power == 1 else diff.square()
    if horizon_weights is not None:
        weights = torch.as_tensor(horizon_weights, device=loss.device, dtype=loss.dtype)
        if weights.ndim != 1 or weights.numel() != loss.shape[0]:
            raise ValueError('One horizon weight is required per forecast step.')
        if not torch.isfinite(weights).all() or torch.any(weights < 0) or weights.mean() <= 0:
            raise ValueError('Horizon weights must be finite, nonnegative, and have positive mean.')
        loss = loss * (weights / weights.mean()).reshape(-1, *([1] * (loss.ndim - 1)))
    return loss.sum() / valid.sum()


def masked_mae_loss(y_pred, y_true, horizon_weights=None, null_val=None):
    return _reduce(y_pred, y_true, 1, horizon_weights, null_val)


def masked_mse_loss(y_pred, y_true, horizon_weights=None, null_val=None):
    return _reduce(y_pred, y_true, 2, horizon_weights, null_val)


def masked_mape_loss(y_pred, y_true, eps=1e-5, null_val=None):
    valid = _valid(y_pred, y_true, null_val) & (y_true.abs() > eps)
    if not torch.any(valid):
        return torch.full((), float('nan'), device=y_true.device)
    return ((y_pred[valid] - y_true[valid]).abs() / y_true[valid].abs()).mean()

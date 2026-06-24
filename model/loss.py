import torch


def _build_mask(y_true, null_val=None):
    """Build a metric mask without dropping physically valid zero targets.

    The WeatherBench variables used here can legitimately be zero, especially
    total cloud cover and wind components. Earlier DCRNN-style masked losses
    used ``0`` as a missing-value sentinel; that is not appropriate for these
    fields. By default we therefore include all finite targets and reserve
    explicit masking for a caller-provided null value.
    """
    if null_val is None:
        mask = torch.isfinite(y_true).float()
    elif isinstance(null_val, float) and null_val != null_val:
        mask = torch.isfinite(y_true).float()
    else:
        mask = (y_true != null_val).float()
    mask_mean = mask.mean()
    if mask_mean > 0:
        mask = mask / mask_mean
    return mask


def _sanitize_loss(loss):
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss


def _prepare_horizon_weights(y_true, horizon_weights=None):
    if horizon_weights is None:
        return None

    if not torch.is_tensor(horizon_weights):
        horizon_weights = torch.tensor(horizon_weights, dtype=y_true.dtype, device=y_true.device)
    else:
        horizon_weights = horizon_weights.to(device=y_true.device, dtype=y_true.dtype)

    if horizon_weights.dim() != 1:
        raise ValueError("horizon_weights must be a 1D tensor or list.")
    if horizon_weights.numel() != y_true.shape[0]:
        raise ValueError(
            f"Expected {y_true.shape[0]} horizon weights, got {horizon_weights.numel()}."
        )

    horizon_weights = horizon_weights / horizon_weights.mean().clamp_min(1e-8)
    return horizon_weights.view(-1, 1, 1, 1)


def masked_mae_loss(y_pred, y_true, horizon_weights=None, null_val=None):
    mask = _build_mask(y_true, null_val=null_val)
    loss = torch.abs(y_pred - y_true) * mask
    weights = _prepare_horizon_weights(y_true, horizon_weights)
    if weights is not None:
        loss = loss * weights
    return _sanitize_loss(loss).mean()


def masked_mse_loss(y_pred, y_true, horizon_weights=None, null_val=None):
    mask = _build_mask(y_true, null_val=null_val)
    loss = torch.square(y_pred - y_true) * mask
    weights = _prepare_horizon_weights(y_true, horizon_weights)
    if weights is not None:
        loss = loss * weights
    return _sanitize_loss(loss).mean()


def masked_mape_loss(y_pred, y_true, eps=1.0e-5, null_val=None):
    mask = _build_mask(y_true, null_val=null_val)
    mask = mask * (torch.abs(y_true) > eps).float()
    loss = torch.abs(y_pred - y_true) / torch.abs(y_true).clamp_min(eps)
    loss = loss * mask
    return _sanitize_loss(loss).abs().mean()

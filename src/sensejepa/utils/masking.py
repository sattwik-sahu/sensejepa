import torch


def mask_tokens(
    tokens: torch.Tensor, p_mask: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masks a fixed number of sensor tokens per batch element.

    For each example in the batch, randomly selects a subset of sensors to mask
    based on p_mask. Ensures that every batch element retains the same number
    of unmasked tokens.

    Args:
        tokens (torch.Tensor): Input tensor of shape (batch_size, n_sensors, dim).
        p_mask (float): Probability of masking a sensor (0.0 to 1.0).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - x_masked: The subset of unmasked tokens (batch_size, n_keep, dim).
            - mask: Binary tensor (batch_size, n_sensors) where 1 indicates masked
              and 0 indicates kept.
    """
    batch_size, n_sensors, dim = tokens.shape
    n_keep = int(n_sensors * (1 - p_mask))

    # 1. Generate random noise for each sensor position
    # noise shape: (batch_size, n_sensors)
    noise = torch.rand(batch_size, n_sensors, device=tokens.device)

    # 2. Sort noise to get random indices
    # ids_keep will contain the indices of the sensors we want to keep
    ids_shuffle = torch.argsort(noise, dim=1)  # (batch_size, n_sensors)
    ids_keep = ids_shuffle[:, :n_keep]  # (batch_size, n_keep)

    # 3. Gather the kept tokens
    # We need to expand ids_keep to match the 'dim' dimension
    # shape: (batch_size, n_keep, dim)
    ids_keep_ext = ids_keep.unsqueeze(-1).repeat(1, 1, dim)
    x_masked = torch.gather(tokens, dim=1, index=ids_keep_ext)

    # 4. Generate the binary mask tensor (0 is kept, 1 is masked)
    mask = torch.ones([batch_size, n_sensors], device=tokens.device)
    mask[:, :n_keep] = 0

    # Unshuffle the mask so it corresponds to the original sensor order
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask

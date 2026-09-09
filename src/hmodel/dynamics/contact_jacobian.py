import torch


def contact_jacobian(
    body_jac: torch.Tensor,
    body_indices: list[int],
) -> torch.Tensor:
    """Extract the linear-velocity sub-Jacobian for contact bodies.

    The full body Jacobian maps generalised velocities to *spatial* (6D)
    velocities.  For contact constraints only the linear velocity matters.
    This function picks the bottom-3 rows (linear part) for the requested
    contact bodies and stacks them into a single matrix J_c.

    Args:
        body_jac:     (..., J, 6, N) body Jacobian from :func:`body_jacobian`.
        body_indices: Indices of contact bodies (e.g. ``[7, 8]`` for SMPL feet).

    Returns:
        (..., 3 * n_c, N) contact Jacobian, n_c = len(body_indices).
    """
    # rows 3:6 of the 6D Jacobian are the linear-velocity rows
    Jc = body_jac[..., body_indices, 3:, :]  # (..., n_c, 3, N)
    batch = body_jac.shape[:-3]
    N = body_jac.shape[-1]
    n_c = len(body_indices)
    return Jc.reshape(*batch, 3 * n_c, N)

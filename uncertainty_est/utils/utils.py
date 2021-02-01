import torch

from scipy.integrate import trapz


def to_np(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def estimate_normalizing_constant(
    density_func, interval=(-10, 10), samples=1000, device="cpu"
):
    """
    Numericall integrate a EBM in the specified interval. Only works for 2D datasets.
    """
    interp = torch.linspace(*interval, samples)
    x, y = torch.meshgrid(interp, interp)
    grid = torch.stack((x.reshape(-1), y.reshape(-1)), 1)
    p_x = density_func(grid.to(device))

    dx = (abs(interval[0]) + abs(interval[1])) / samples
    Z = trapz(trapz(to_np(p_x).reshape(len(interp), len(interp)), dx=dx), dx=dx)

    return torch.tensor(Z)

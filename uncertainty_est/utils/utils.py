import torch
from tqdm import tqdm
from scipy.integrate import trapz


def to_np(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def estimate_normalizing_constant(
    density_func,
    interval=(-10, 10),
    num_samples=1000,
    device="cpu",
    dimensions=2,
    batch_size=100_000,
):
    """
    Numerically integrate a funtion in the specified interval.
    """
    interp = torch.linspace(*interval, num_samples)
    grid_coords = torch.meshgrid(*[interp for _ in range(dimensions)])
    grid = torch.stack([coords.reshape(-1) for coords in grid_coords], 1)

    p_x = []
    for samples in tqdm(torch.split(grid, batch_size)):
        p_x.append(density_func(samples.to(device)))
    p_x = torch.cat(p_x)

    dx = (abs(interval[0]) + abs(interval[1])) / num_samples
    # Integrate one dimension after another
    grid_vals = to_np(p_x).reshape(*[len(interp) for _ in range(dimensions)])
    for _ in range(dimensions):
        grid_vals = trapz(grid_vals, dx=dx, axis=-1)

    return torch.tensor(grid_vals)


if __name__ == "__main__":
    dims = 2
    samples = 100
    print(
        estimate_normalizing_constant(
            lambda x: torch.empty(x.shape[0]).fill_(1 / (samples ** dims)),
            num_samples=samples,
            dimensions=dims,
        )
    )

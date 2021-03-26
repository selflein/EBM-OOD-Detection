import torch
from tqdm import tqdm
from scipy.integrate import trapz


def to_np(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def eval_func_on_grid(
    density_func,
    interval=(-10, 10),
    num_samples=200,
    device="cpu",
    dimensions=2,
    batch_size=10_000,
    dtype=torch.float32,
):
    interp = torch.linspace(*interval, num_samples)
    grid_coords = torch.meshgrid(*[interp for _ in range(dimensions)])
    grid = torch.stack([coords.reshape(-1) for coords in grid_coords], 1).to(dtype)

    vals = []
    for samples in tqdm(torch.split(grid, batch_size)):
        vals.append(density_func(samples.to(device)).cpu())
    vals = torch.cat(vals)
    return grid_coords, vals


def estimate_normalizing_constant(
    density_func,
    interval=(-10, 10),
    num_samples=200,
    device="cpu",
    dimensions=2,
    batch_size=10_000,
    dtype=torch.float32,
):
    """
    Numerically integrate a funtion in the specified interval.
    """
    with torch.no_grad():
        _, p_x = eval_func_on_grid(
            density_func, interval, num_samples, device, dimensions, batch_size, dtype
        )

        dx = (abs(interval[0]) + abs(interval[1])) / num_samples
        # Integrate one dimension after another
        grid_vals = to_np(p_x).reshape(*[num_samples for _ in range(dimensions)])
        for _ in range(dimensions):
            grid_vals = trapz(grid_vals, dx=dx, axis=-1)

        return torch.tensor(grid_vals)


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def split_leading_dim(x, shape):
    """Reshapes the leading dim of `x` to have the given shape."""
    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)


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

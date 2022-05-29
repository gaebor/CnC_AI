from torch import load, searchsorted, rand


def number_of_digits(n):
    return len(str(n))


def retrieve(t):
    return t.detach().to('cpu')


def max(m, axes):
    for axis in axes:
        m = m.max(dim=axis).values
    return m


def get_log_formatter(indices):
    return ', '.join(
        f'{index_name}: {{:0{len(str(max_value))}d}}/{max_value}'
        for index_name, max_value in indices.items()
    )


def torch_safe_load(filename, constructor):
    try:
        return load(filename)
    except FileNotFoundError:
        return constructor()


def multi_sample(p):
    return searchsorted(p.cumsum(axis=1), rand(p.shape[0], 1, dtype=p.dtype, device=p.device))[
        :, 0
    ]


def dictmap(d, f):
    return {k: f(v) for k, v in d.items()}

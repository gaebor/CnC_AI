def number_of_digits(n):
    return len(str(n))


def retrieve(t):
    return t.detach().to('cpu').numpy()


def max(m, axes):
    for axis in axes:
        m = m.max(dim=axis).values
    return m


def get_log_formatter(indices):
    return ', '.join(
        f'{index_name}: {{:0{len(str(max_value))}d}}/{max_value}'
        for index_name, max_value in indices.items()
    )


def dictmap(d, f):
    return {k: f(v) for k, v in d.items()}

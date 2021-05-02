def number_of_digits(n):
    return len(str(n))


def retrieve(t):
    return t.detach().to('cpu')


def max(m, axes):
    for axis in axes:
        m = m.max(dim=axis).values
    return m

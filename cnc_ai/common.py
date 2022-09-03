import matplotlib.pyplot as plt
import numpy

plt.ion()
plt.show()


def plot_images(*sets_of_images):
    shape_x = max([images[0].shape[0] for images in sets_of_images])
    shape_y = max([images[0].shape[1] for images in sets_of_images])
    big_image = numpy.zeros((shape_x, len(sets_of_images[0]) * shape_y), dtype=float)
    for images in sets_of_images:
        for i, image in enumerate(images):
            big_image[: image.shape[0], i * shape_y : i * shape_y + image.shape[1]] = image

    plt.gcf().clear()
    plt.imshow(big_image)
    plt.pause(0.1)


def number_of_digits(n):
    return len(str(n))


def retrieve(t):
    return t.detach().to('cpu').numpy()


def get_log_formatter(indices):
    return ', '.join(
        f'{index_name}: {{:0{len(str(max_value))}d}}/{max_value}'
        for index_name, max_value in indices.items()
    )


def dictmap(d, f):
    return {k: f(v) for k, v in d.items()}

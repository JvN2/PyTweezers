import numpy as np


def get_subarray(array, center, width):
    x_center, y_center = center
    half_width = width // 2

    x_start = max(0, x_center - half_width)
    y_start = max(0, y_center - half_width)
    x_end = min(array.shape[0], x_center + half_width + 1)
    y_end = min(array.shape[1], y_center + half_width + 1)

    if x_start == 0:
        x_end = min(array.shape[0], width)
    elif x_end == array.shape[0]:
        x_start = max(0, array.shape[0] - width)

    if y_start == 0:
        y_end = min(array.shape[1], width)
    elif y_end == array.shape[1]:
        y_start = max(0, array.shape[1] - width)

    return array[x_start:x_end, y_start:y_end], [
        x_start + width // 2,
        y_start + width // 2,
    ]


# Example usage:
array = np.random.rand(10, 10)
array = np.array([[x for x in range(10)] for y in range(10)])
center = (0, 100)
width = 5
subarray, center = get_subarray(array, center, width)
print(subarray)
print(center)

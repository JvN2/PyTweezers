import numpy as np
from icecream import ic


def get_subarray(array: np.ndarray, center: tuple, size: int) -> np.ndarray:
    height, width = array.shape[:2]
    half_size = size // 2
    x, y = center

    # Calculate the coordinates of the top-left and bottom-right corners
    x0, x1 = x - half_size, x + half_size
    y0, y1 = y - half_size, y + half_size

    # Adjust the coordinates if they exceed the image boundaries
    if x0 < 0:
        x0 = 0
        x1 = size
    if x1 > width:
        x1 = width
        x0 = width - size
    if y0 < 0:
        y0 = 0
        y1 = size
    if y1 > height:
        y1 = height
        y0 = height - size

    # Ensure the coordinates are within the image boundaries
    x0 = max(0, x0)
    x1 = min(width, x1)
    y0 = max(0, y0)
    y1 = min(height, y1)

    # Extract the square sub-array from the image
    if len (array.shape) == 3:
        sub_array = array[y0:y1, x0:x1, :]
    else:
       sub_array = array[y0:y1, x0:x1]

    return sub_array, [
        (x0 + x1) // 2,
        (y0 + y1) // 2,
    ]


# Example usage:
array = np.random.rand(1000, 1000)
array = np.array([[x for x in range(1000)] for y in range(1000)])
center = (500, 1500)
width = 10
subarray, center = get_subarray(array.T, center, width)
ic(subarray)
ic(center)
ic(subarray.shape)

import cv2
import numpy as np

import numpy as np

def extract_square(image: np.ndarray, center: tuple, width: int) -> np.ndarray:
    height, img_width = image.shape[:2]
    half_width = width // 2
    x, y = center

    # Calculate the coordinates of the top-left and bottom-right corners
    x0, x1 = x - half_width, x + half_width
    y0, y1 = y - half_width, y + half_width

    # Adjust the coordinates if they exceed the image boundaries
    if x0 < 0:
        x0 = 0
        x1 = width
    if x1 > img_width:
        x1 = img_width
        x0 = img_width - width
    if y0 < 0:
        y0 = 0
        y1 = width
    if y1 > height:
        y1 = height
        y0 = height - width

    # Ensure the coordinates are within the image boundaries
    x0 = max(0, x0)
    x1 = min(img_width, x1)
    y0 = max(0, y0)
    y1 = min(height, y1)

    # Extract the square region from the image
    square = image[y0:y1, x0:x1]

    return square, (x0, y0)

# Example usage
image = np.random.rand(1000, 1000, 3).astype(np.uint8)
center = (0, 50000)
width = 500
square , offset= extract_square(image, center, width)
print(square.shape, offset)


import cv2
import numpy as np
import matplotlib.pyplot as plt

from ImageProcessing import load_bin_file
from MT_Tracker import bandpass_filter

# Load the image
# image = cv2.imread("path_to_your_image", cv2.IMREAD_GRAYSCALE)
filename = r"d:\users\noort\data\20241212\data_006.hdf"
frames = load_bin_file(filename)
image = frames[-10]

# Perform FFT
fft = np.fft.fft2(image)
fft_shifted = np.fft.fftshift(fft)

# Apply a high-pass filter to remove low frequencies
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Create a mask first, center square is 1, remaining all zero
mask = np.ones((rows, cols), np.uint8)
r = 30  # radius of the high-pass filter
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) <= r
mask[mask_area] = 0

mask = bandpass_filter(image, high=8, low=25, centered=True, cut_offset=True)

# Apply mask and inverse FFT
fft_shifted_masked = fft_shifted * mask
inverse_fft = np.fft.ifftshift(fft_shifted_masked)
reconstructed_image = np.fft.ifft2(inverse_fft)
reconstructed_image = np.abs(reconstructed_image)

# Find the coordinates of the maximum value in the reconstructed image
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(reconstructed_image)
center_coordinates = max_loc

print(f"The center coordinates of the circular object are: {center_coordinates}")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# Display the original and reconstructed images
plt.subplot(121), plt.imshow(image, cmap="gray")
# plt.scatter(*center_coordinates, c="r")
circle = plt.Circle(center_coordinates, 15, color="r", fill=False)
axes[0].add_artist(circle)

plt.title("Input Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(reconstructed_image, cmap="gray")
plt.title("Reconstructed Image"), plt.xticks([]), plt.yticks([])
plt.show()

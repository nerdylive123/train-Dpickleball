import cv2
import numpy as np

# Create a dummy image for demonstration
image = np.zeros((100, 100, 3), dtype=np.uint8)
image[:, :, 0] = 255  # Blue channel

# Attempt to save the image
cv2.imwrite("output_image.png", image)
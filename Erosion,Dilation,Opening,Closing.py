from google.colab import drive
drive.mount('/content/drive')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Load the damaged digital photo image
image = cv2.imread('/content/drive/MyDrive/cv/image.jpg', cv2.IMREAD_GRAYSCALE)

# Define the kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Erosion
erosion = cv2.erode(image, kernel, iterations=1)

# Dilation
dilation = cv2.dilate(image, kernel, iterations=1)

# Opening (erosion followed by dilation)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Closing (dilation followed by erosion)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Display the results
plt.figure(figsize=(14, 10))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Erosion")
plt.imshow(erosion, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Dilation")
plt.imshow(dilation, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Opening")
plt.imshow(opening, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Closing")
plt.imshow(closing, cmap='gray')
plt.axis('off')

plt.show()

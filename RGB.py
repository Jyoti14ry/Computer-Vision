from google.colab import drive
drive.mount('/content/drive')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Load the original image
original_image = cv2.imread('/content/drive/MyDrive/cv/image.jpg')
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

# Convert to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Perform basic thresholding for segmentation
def segment_image(gray_image, threshold_value):
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image

# Set the threshold value for segmentation
threshold_value = 127

# Segment the image
binary_image = segment_image(gray_image, threshold_value)

# Apply Canny edge detection
def apply_canny_edge_detection(binary_image, low_threshold, high_threshold):
    edges = cv2.Canny(binary_image, low_threshold, high_threshold)
    return edges

# Set thresholds for Canny edge detection
low_threshold = 50
high_threshold = 150

# Apply Canny edge detection
edges = apply_canny_edge_detection(binary_image, low_threshold, high_threshold)

# Display the results
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(original_image_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Segmented Binary Image")
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Edges Detected with Canny")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()

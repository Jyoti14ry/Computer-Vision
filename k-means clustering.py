import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_image_kmeans(image_path, K):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    # Convert the image from BGR (OpenCV default) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image into a 2D array of pixels
    pixel_values = image_rgb.reshape((-1, 3))
    # Convert to float type
    pixel_values = np.float32(pixel_values)

    # Define criteria for the K-means algorithm
    # (type, max_iter, epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Apply K-means clustering to segment the image
    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert the centers to uint8 (as image pixels are integers)
    centers = np.uint8(centers)

    # Map the labels to the corresponding cluster centers
    segmented_image = centers[labels.flatten()]

    # Reshape the segmented image to the original image shape
    segmented_image = segmented_image.reshape(image_rgb.shape)

    # Display the original and segmented images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title('Segmented Image with K = {}'.format(K))
    plt.axis('off')

    plt.show()

# Path to the image of the fruit basket
image_path = '/content/fruit-basket2.jpg'  # Replace with the path to your image file
K = 10  # Number of clusters (e.g., 4 for different types of fruits)

# Call the function to perform segmentation
segment_image_kmeans(image_path, K)

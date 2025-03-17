from google.colab import drive
drive.mount('/content/drive')
import cv2
import numpy as np
import matplotlib.pyplot as plt
file_path = '/content/drive/MyDrive/cv/image.jpg'

# Load a color image
image = cv2.imread('passport size photo.jpg')

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image. Please check the file path and try again.")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Define the Laplacian filter
    def laplacian_filter(image):
        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=np.float32)
        # Apply the Laplacian filter
        laplacian_filtered = cv2.filter2D(image, -1, laplacian_kernel)
        return laplacian_filtered

    # Sharpen the image
    def sharpen_image(image, alpha=1.0):
        laplacian = laplacian_filter(image)
        sharpened_image = cv2.addWeighted(image, 1, laplacian, alpha, 0)
        return sharpened_image

    # Apply the sharpening
    sharpened_image = sharpen_image(image_rgb, alpha=1.0)

    # Display the results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Sharpened Image")
    plt.imshow(sharpened_image)
    plt.axis('off')

    plt.show()

from google.colab import drive
drive.mount('/content/drive')
import cv2
from matplotlib import pyplot as plt
# Function to display images
def display_image(title, image, cmap=None):
 plt.imshow(image,cmap=cmap)
 plt.title(title)
 plt.axis()
 plt.show()
# a. To read an image
file_path = '/content/drive/MyDrive/CV lab/image.jpg'
image = cv2.imread(file_path)
## b. To show an image
def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# c. Convert RGB to Gray Scale
def rgb_to_gray(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image
# d. Read RGB values of a pixel
def get_pixel_value(image, x, y):
    (b, g, r) = image[y, x]
    return (r, g, b)
# e. Convert Gray Scale to Binary
def gray_to_binary(gray_image, threshold=128):
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image
# f. Perform Image Crop
def crop_image(image, x, y, w, h):
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image
# g. Perform Image Resize
def resize_image(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    return resized_image
# h. Perform Image Rotation
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image
# i. Histogram Equalization
def histogram_equalization(gray_image):
    equalized_image = cv2.equalizeHist(gray_image)
    return equalized_image

# Import libraies
import os
import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Initialize the path of given image
image_path = "res\ggiven_image.png"


def read_images_from_dir(image_path):
    """
    This function will read images from given path
    :param image_path: THe path of the given image
    :return: images
    """
    if os.path.exists(image_path):
        # Now read the image
        image = cv2.imread(image_path)
        return image
    else:
        raise ValueError("Given image path does not exist")


def process_image(img):
    """
    This function will process the image
    :param img: The given image
    :return: processed image
    """
    # Firstly convert the given image to GrayScale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Now create another blurred version of the images
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # Now apply the canny method to find out the edge
    edge = cv2.canny(blurred, 50, 150)
    # Now finally return the edge image
    return edge


# Let's extract the axes from the edge image
def extract_axes(edge):
    """
    This function will extract the axes from the edge image
    :param edge: The given edge image
    :return: axes
    """
    # To extract the line from edge image we need to apply the HoughLine
    line = cv2.HoughLinesP(edge, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    if line is None:
        return None
    else:
        # Now we can extract the axes
        return line



# Now we will find out the ploting points from image, lines
def extract_plot_points(img, lines):
    """
    This function will extract the ploting points from the image
    :param img: The given image
    :param lines: The given lines
    :return: ploting points
    """
    # Fistly, we need to convert the img to binary image
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # Get the points of the coordinate from that binary images
    points = np.column_stack(np.where(binary_img > 0))
    y_values = points[:,0]
    x_values = points[:,1]
    # Now focus on finding the peaks and troughs
    peaks, _ = find_peaks(y_values)
    troughs, _ = find_peaks(-y_values)
    ey_points = []
    for peak in peaks:
        key_points.append((x_values[peak], y_values[peak]))
    for trough in troughs:
        key_points.append((x_values[trough], y_values[trough]))

    return key_points




def main():
    """
    This is the main function of this program :
    :return:
    """
    # Preprocess the image to detect edges
    img = read_images_from_dir(image_path)
    edges = process_image(img)

    # Extract axes (though in this case we'll focus on extracting the plot points)
    lines = extract_axes(edges)
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    key_points = extract_plot_points(original_image, lines)
    for idx, point in enumerate(key_points):
        print(f"Key Point {idx + 1}: (X: {point[0]}, Y: {point[1]})")

    # Ploting
    plt.imshow(original_image, cmap='gray')
    for point in key_points:
        plt.plot(point[0], point[1], 'ro')  # Mark key points in red
    plt.show()


if __name__ == "__main__":
    main()

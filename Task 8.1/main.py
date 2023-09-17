# Importing necessary Libraries and Modules
import numpy as np
import cv2 as cv


# Function to implement Masking (Retrieved from Google Collab from Session Resources)
def color_detection(img, lower_bound, upper_bound):
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    first = img[:, :, 0]
    second = img[:, :, 1]
    third = img[:, :, 2]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if ((first[i, j] >= lower_bound[0] and first[i, j] <= upper_bound[0]) and
                    (second[i, j] >= lower_bound[1] and second[i, j] <= upper_bound[1]) and
                    (third[i, j] >= lower_bound[2] and third[i, j] <= upper_bound[2])):
                mask[i, j] = 255
            else:
                mask[i, j] = 0
    return mask


# Reading Image
img_name = 'rb_021.jpg'
img = cv.imread(img_name)

# Applying Gaussian Filter to Image with a kernel of 9x9 matrix
blur = cv.GaussianBlur(img, (9, 9), 0)

# Color Ranges needed for Masking
lower_dark_blue = [110, 150, 0]
upper_dark_blue = [125, 255, 255]

lower_red = [0, 150, 0]
upper_red = [5, 255, 255]

# Converting Image from BGR to HSV (Better for Color Detection) to implement Masking
HSV_img = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
mask_blue = color_detection(HSV_img, lower_dark_blue, upper_dark_blue)
mask_red = color_detection(HSV_img, lower_red, upper_red)

# Implementing Opening (Erosion then Dilation) to filter unwanted Noise
kernel = np.ones((3, 3), np.uint8)
opening_blue = cv.morphologyEx(mask_blue, cv.MORPH_OPEN, kernel)
# opening_red = cv.morphologyEx(mask_red, cv.MORPH_OPEN, kernel)

# Finding Contours in Blue Mask Image & Red Mask Image
contours_blue, hierarchy_blue = cv.findContours(opening_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours_red, hierarchy_red = cv.findContours(mask_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Drawing an enclosing circle around blue ball to show detection
max_center = 0
max_radius = 0
for contour in contours_blue:
    (x, y), radius = cv.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    if radius > max_radius:
        max_radius = radius
        max_center = center
# To avoid noise
if max_radius > 10:
    cv.circle(img, max_center, max_radius, (0, 255, 0), 2)

# Drawing an enclosing circle around Red ball to show detection
if img_name[0] != 'b':
    max_center = 0
    max_radius = 0
    for contour in contours_red:
        (x, y), radius = cv.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        if radius > max_radius:
            max_radius = radius
            max_center = center
    # To avoid noise
    if max_radius > 10:
        cv.circle(img, max_center, max_radius, (0, 255, 0), 2)

# Showing Image
cv.imshow("Image", img)
# cv.imshow("Filter Image", blur)
cv.imshow("Mask Blue", mask_blue)
cv.imshow("Mask Red", mask_red)
cv.waitKey(0)
cv.destroyAllWindows()

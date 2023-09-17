import numpy as np
import cv2 as cv


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


img_name = 'rb_006.jpg'
img = cv.imread(img_name)

blur = cv.GaussianBlur(img, (9, 9), 0)

lower_light_blue = [105, 150, 0]
upper_light_blue = [110, 255, 255]

lower_dark_blue = [110, 150, 0]
upper_dark_blue = [125, 255, 255]

lower_red = [0, 150, 0]
upper_red = [5, 255, 255]

HSV_img = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
# mask_light = color_detection(HSV_img, lower_light_blue, upper_light_blue)
mask_blue = color_detection(HSV_img, lower_dark_blue, upper_dark_blue)
# mask_blue = mask_blue + mask_light
mask_red = color_detection(HSV_img, lower_red, upper_red)

kernel = np.ones((3, 3), np.uint8)
opening_blue = cv.morphologyEx(mask_blue, cv.MORPH_OPEN, kernel)
# opening_red = cv.morphologyEx(mask_red, cv.MORPH_OPEN, kernel)

contours_blue, hierarchy_blue = cv.findContours(opening_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours_red, hierarchy_red = cv.findContours(mask_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

max_center = 0
max_radius = 0
for contour in contours_blue:
    (x, y), radius = cv.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    if radius > max_radius:
        max_radius = radius
        max_center = center

if max_radius > 10:
    cv.circle(img, max_center, max_radius, (0, 255, 0), 2)

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

    if max_radius > 10:
        cv.circle(img, max_center, max_radius, (0, 255, 0), 2)


cv.imshow("Image", img)
# cv.imshow("Filter Image", blur)
cv.imshow("Mask Blue", mask_blue)
cv.imshow("Mask Red", mask_red)
cv.waitKey(0)
cv.destroyAllWindows()

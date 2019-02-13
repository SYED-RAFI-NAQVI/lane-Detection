import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def canny(image):
    grayImg1 = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(grayImg1, (5,5), 0)
    canny = cv.Canny(blur, 50,150)
    return canny


def region_of_interest(image):
    height = img.shape[0]
    triangle = np.array([[(200, height), (1100, height), (550,250)]])
    mask = np.zeros_like(image)
    cv.fillPoly(mask, triangle, 255)
    masked_image = cv.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(line_image, (x1,y1), (x2,y2), (0,255,0), 10)
    return line_image


img =  cv.imread('test_image.jpg')
grayImg = np.copy(img)
canny = canny(grayImg)
roi = region_of_interest(canny)
lines = cv.HoughLinesP(roi, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5 )
line_image = display_lines(grayImg, lines)
added_image = cv.addWeighted(grayImg, 0.8, line_image, 1, 1)


cv.imshow('canny', canny)
cv.imshow('roi', roi)
cv.imshow('line_image', line_image)
cv.imshow('Added', added_image)
cv.waitKey(0)
cv.destroyAllWindows()
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def make_coordinate(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])
    
    
    
def average_lines(image, lines):
    right = []
    left = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    left_line = make_coordinate(image, left_avg)
    right_line = make_coordinate(image, right_avg)
    return np.array([left_line, right_line])



def canny(image):
    grayImg1 = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(grayImg1, (5,5), 0)
    canny = cv.Canny(blur, 50,150)
    return canny


def region_of_interest(image):
    height = image.shape[0]
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
averaged_lines = average_lines(grayImg, lines)
line_image = display_lines(grayImg, averaged_lines)
added_image = cv.addWeighted(grayImg, 0.8, line_image, 1, 1)
cv.imshow('canny', canny)
cv.imshow('roi', roi)
cv.imshow('line_image', line_image)
cv.imshow('Added', added_image)
cv.waitKey(0)
cv.destroyAllWindows()
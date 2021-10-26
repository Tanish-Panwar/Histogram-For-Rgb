import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# img = cv.imread('imgs/pic.jpg')
img = cv.imread('imgs/pic.jpg')
cv.imshow("IMG", img)
# Just resizing the img....
# def rescaleFrame(frame,scale=0.65):
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)
#     dimensions = (width, height)
#     return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# short = rescaleFrame(img)

# MAde a blank image with numpy .zeros....
blank = np.zeros(img.shape[:2],dtype='uint8')

# # Setted a grayscale image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("GRAY", gray)

# # Made a cicle
circle = cv.circle(blank, (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)
# cv.imshow("circle", circle)

# # Made a mask
mask = cv.bitwise_and(gray, gray, mask=circle)
# cv.imshow("mask", mask)


# # Computing Histogram in Grayscale Image...


# # COMPUTING...HISTOGRAM...
# gray_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])
# plt.figure()
# plt.title("GRAYSCALE HISTOGRAM")
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(gray_hist)
# plt.xlim([0, 256])
# plt.show()








plt.figure()
plt.title("COLOR HISTOGRAM")
plt.xlabel('Bins')
plt.ylabel('# of pixels')

# COMPUTING RGB COLOR HISTOGRAM....
colors = ('b', 'g', 'r')
for i,col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0,256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])
plt.show()





cv.waitKey(0)
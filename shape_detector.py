import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import imutils

img = cv.imread('test2.jpg', 0)
img2=img.copy()
# Размытие изображения по краям
# img = cv.medianBlur(img, 5)
img = cv.GaussianBlur(img, (7,7), 0)
# Бинаризация по пороговым значениям
ret, th1 = cv.threshold(img, 140, 255, cv.THRESH_BINARY)
# Инициализация ядра
kernel = np.ones((3, 3), np.uint8)
#
opening = cv.morphologyEx(th1, cv.MORPH_OPEN, kernel)

cnts = cv.findContours(opening, cv.RETR_LIST,
                       cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    # Obtain bounding rectangle to get measurements
    x, y, w, h = cv.boundingRect(c)

    # Find centroid
    M = cv.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Draw the contour and center of the shape on the image
    cv.rectangle(img2, (x, y), (x + w, y + h), (255, 255, 255), 5)
    sv = cv.circle(img2, (cX, cY), 11, (320, 159, 22), -1)
    # cv.drawContours(sv, [c], -1, (0, 255, 0), 2)
cv.imwrite('cnts.png', img2)
# Массив названий изображений
titles = ['Original Image', 'Global Thresholding (v = 127)', 'Morphology 1', 'Morphology 2']
# Массив изображений
images = [img, th1, opening, img2]
for i in range(4):
    # Вывод изображения на график
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    # Получение названия изображения
    plt.title(titles[i])
    # Отобразить оси <x,y>
    plt.xticks([]), plt.yticks([])
plt.show()

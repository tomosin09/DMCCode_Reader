import cv2 as cv
import numpy as np
from numpy.linalg import lstsq
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN


class DPM_Recovery:
    def __init__(self, image, temp):
        # Image for restoration DPM DMC
        self.image = image
        # Correlation search pattern
        self.temp = temp
        self.img_rgb = cv.imread(self.image)
        # gray-scale conversion
        self.img_gray = cv.cvtColor(self.img_rgb,
                                    cv.COLOR_BGR2GRAY)
        self.template = cv.imread(self.temp, 0)

        self.groups_points = self.img_rgb.copy()
        self.pts = []
        # Initialization array nearest points from line
        self.nearest_points = []
        self.lines = []
        self.intersections = []

    def binarization(self):
        # search for matches
        res = cv.matchTemplate(self.img_gray, self.template, cv.TM_CCOEFF_NORMED)
        ret, self.thr = cv.threshold(res, 0.55, 255, cv.THRESH_BINARY)
        self.thr = self.thr.astype('uint8')
        cnts = cv.findContours(self.thr, cv.RETR_LIST,
                               cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            # Obtain bounding rectangle to get measurements
            x, y, w, h = cv.boundingRect(c)
            # Find centroid
            M = cv.moments(c)
            # Check division by zero
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            self.pts.append([cX, cY])
            # Draw the contour and center of the shape on the image
            sv = cv.circle(self.thr, (cX, cY),
                           7, (255, 255, 255), -1)

    def find_point_groups(self):
        points = np.array(self.pts)
        db = DBSCAN(eps=32, min_samples=3).fit(points)
        # indices of found clusters
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # n_noise_ = list(labels).count(-1)
        # print('Estimated number of clusters: %d' % n_clusters_)
        # print('Estimated number of noise points: %d' % n_noise_)
        unique_labels = set(labels)
        self.clusters = [[] for i in unique_labels]
        for i, j in zip(points, labels):
            if j != -1:
                self.clusters[j].append(i)
        colors = np.random.randint(30, 255, (len(self.clusters), 3))
        for i in range(len(self.clusters)):
            i += 1
            for x in self.clusters[i - 1]:
                cv.circle(self.groups_points, (x[0], x[1]), 5,
                          (int(colors[i - 1][0]), int(colors[i - 1][1]), int(colors[i - 1][2])),
                          -1)

    def det_dist_to_points(self, p0, p1, point):
        # Convert to vector
        line_vec = p1 - p0
        pnt_vec = point - p0
        # Find length vector
        line_len = np.linalg.norm(line_vec)
        # Find unit vector
        line_unitvec = [x / line_len for x in line_vec]
        # Scale vector of point by length vector of line
        pnt_vec_scaled = [x * (1 / line_len) for x in pnt_vec]
        # Find the distance to the perpendicular
        t = np.dot(line_unitvec, pnt_vec_scaled)
        # Clamp 't' to the range 0 to 1.
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        # Scale the line vector by 't' to find the nearest location
        nearest = [x * t for x in line_vec]
        # Find the distance between the nearest vector and point vector
        dist = np.linalg.norm(nearest - pnt_vec)
        return dist

    def get_min_dist(self, dist, point, exp):
        return point if dist < exp else None

    def find_nearest_points(self):
        # Loop to iterate over clusters
        for n_cl, cluster in enumerate(self.clusters):
            # If len cluster is 0
            if len(cluster) == 0:
                continue
            elif len(cluster) > 80:
                rect = cv.minAreaRect(np.float32(cluster))
                box = cv.boxPoints(rect)
                box = np.int0(box)
                # Loop to iterate over lines in contour
                for i, item in enumerate(box[::-1]):
                    for c in cluster:
                        if i == 3:
                            dist = self.det_dist_to_points(box[i], box[0], c)
                            point = self.get_min_dist(dist, c, 15)
                            if point is not None:
                                self.nearest_points.append([n_cl + 1, i + 1, point])
                        else:
                            dist = self.det_dist_to_points(box[i], box[i + 1], c)
                            point = self.get_min_dist(dist, c, 15)
                            if point is not None:
                                self.nearest_points.append([n_cl + 1, i + 1, point])
                # cv.drawContours(img3, [box], 0, (255, 0, 255), 2)

    def defining_code_boundaries(self):
        count_intersection = 0
        self.find_nearest_points()
        for p in self.nearest_points:
            for j in self.nearest_points:
                condition = np.sum(p[2] - j[2])
                if condition != 0 and p[1] == j[1] and p[0] == j[0]:
                    line = (p[2], j[2])
                    self.lines.append([p[0], p[1], line])
        for line in self.lines:
            count_intersection = 0
            for point in self.nearest_points:
                if line[0] == point[0] and line[1] == point[1]:
                    dist = self.det_dist_to_points(line[2][0], line[2][1], point[2])
                    point_intersection = self.get_min_dist(dist, point[2], 5)
                    if point_intersection is not None:
                        count_intersection += 1
            self.intersections.append([line[0], line[1], line[2], count_intersection])

    def visualization(self):
        titles = ['Original', 'Binarization', 'Detected groups']
        images = [self.img_rgb, self.thr, self.groups_points]
        for i in range(3):
            plt.subplot(2, 2, i + 1);
            plt.imshow(images[i])
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

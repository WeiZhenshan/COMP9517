# Task1 Hint: (with sample code for the SIFT detector)
# Initialize SIFT detector, detect keypoints, store and show SIFT keypoints of original image in a Numpy array
# Define parameters for SIFT initializations such that we find only 10% of keypoints
import cv2
import matplotlib.pyplot as plt
import imutils
from copy import deepcopy

class SiftDetector():
    def __init__(self, norm="L2", params=None):
        self.detector = self.get_detector(params)
        self.norm = norm

    def get_detector(self, params):
        if params is None:
            params = {}
            params["n_features"] = 623      #changed default nfeatures=0 to 623
            params["n_octave_layers"] = 3
            params["contrast_threshold"] = 0.03
            params["edge_threshold"] = 10
            params["sigma"] = 1.6

        detector = cv2.xfeatures2d.SIFT_create(
            nfeatures=params["n_features"],
            nOctaveLayers=params["n_octave_layers"],
            contrastThreshold=params["contrast_threshold"],
            edgeThreshold=params["edge_threshold"],
            sigma=params["sigma"])

        return detector

###############task1###########################

p = SiftDetector("L2",None)
img = cv2.imread('COMP9517_20T2_Lab2_Image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = p.detector
kp,des = sift.detectAndCompute(gray,None)
I = cv2.drawKeypoints(gray, kp, img)
# cv2.imwrite('1_a.jpg', I)     use nfeatures=0. (default params)
cv2.imwrite('1_b.jpg',I)

##################TASK 2######################################################

img1 = cv2.imread('COMP9517_20T2_Lab2_Image.jpg')

scale_percent = 115  # percent of original size
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img1, dim)
cv2.imwrite('2_a.jpg',resized)

img2=deepcopy(resized)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sift = p.detector
kp2,des2 = sift.detectAndCompute(gray2,None)
I2 = cv2.drawKeypoints(gray2, kp2, img2)

cv2.imwrite('2_b.jpg', I2)

# Task 2. c.
# Yes, the keypoints are roughly same of scaled image and original image. SIFT is scale invariant. Scaling does not affect keypoints. Mostly concentrated in the center of image and top.


bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des,des2,k=2)

g=[]
for m,n in matches:
    if m.distance < 0.75*n.distance:
        g.append([m])

g = sorted(g, key=lambda x: x[0].distance)
img3 = cv2.drawMatchesKnn(gray,kp,gray2,kp2,g[:6], img,flags=2)
cv2.imwrite('2_d.jpg',img3)

################TASK3#################################

img4 = cv2.imread('COMP9517_20T2_Lab2_Image.jpg')

rotated_img=imutils.rotate_bound(img4,60)
cv2.imwrite('3_a.jpg',rotated_img)

img5=deepcopy(rotated_img)
gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
sift = p.detector
kp5,des5 = sift.detectAndCompute(gray5,None)
I5 = cv2.drawKeypoints(gray5, kp5, img5)
cv2.imwrite('3_b.jpg',I5)

# Task 3.c.
# Yes, the keypoints are roughly same of rotated image and original image. SIFT is rotation/orientation invariant. Rotation does not affect keypoints. Mostly concentrated in the center of image and top.

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des,des5,k=2)

g = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        g.append([m])

g = sorted(g, key=lambda x: x[0].distance)
img6 = cv2.drawMatchesKnn(gray,kp,gray5,kp5,g[:7],img,flags=2)
cv2.imwrite('3_d.jpg',img6)

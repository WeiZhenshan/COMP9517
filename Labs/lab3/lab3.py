import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift

from PIL import Image

size = 100, 100

img_names = ["shapes.png", "strawberry.png"]
ext_names = ["coins.png", "kiwi.png"]

images = [i for i in img_names]
ext_images = [i for i in ext_names]


def plot_three_images(figure_title, image1, label1,
                      image2, label2, image3, label3):
    fig = plt.figure()
    fig.suptitle(figure_title)

    # Display the first image
    fig.add_subplot(1, 3, 1)
    plt.imshow(image1)
    plt.axis('off')
    plt.title(label1)

    # Display the second image
    fig.add_subplot(1, 3, 2)
    plt.imshow(image2)
    plt.axis('off')
    plt.title(label2)

    # Display the third image
    fig.add_subplot(1, 3, 3)
    plt.imshow(image3)
    plt.axis('off')
    plt.title(label3)

    plt.show()

def MeanShifting(img):
    img_mat = np.array(img)[:, :, :3]

    blueChannel = img_mat[:, :, 0]
    greenChannel = img_mat[:, :, 1]
    redChannel = img_mat[:, :, 2]

    red_img = np.zeros(img_mat.shape)
    green_img = np.zeros(img_mat.shape)
    blue_img = np.zeros(img_mat.shape)

    blue_img[:, :, 0] = blueChannel
    green_img[:, :, 1] = greenChannel
    red_img[:, :, 2] = redChannel

    blue_flatten = blueChannel.flatten()
    green_flatten = greenChannel.flatten()
    red_flatten = redChannel.flatten()

    colour_samples = []
    colour_samples = np.stack((blue_flatten, green_flatten, red_flatten), axis=1)
    # print('color_samples shape', colour_samples.shape)

    ms_clf = MeanShift(bin_seeding=True)
    ms_labels = ms_clf.fit_predict(colour_samples)

    ms_labels = ms_labels.reshape(img_mat.shape[:2])

    # print('ms labels reshape', ms_labels.shape)

    return ms_labels

def WaterShedding(img):
    img1 = img.convert("L")
    img_array = np.array(img1)
    # print(img_array.shape)
    distance = ndi.distance_transform_edt(img_array)
    markers = []
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                labels=img_array)
    markers = ndi.label(local_maxi)[0]

    ws_labels = watershed(-distance, markers, mask=img_array)

    return ws_labels

def WaterShedOptimizeC(img):
    img1 = img.convert("L")
    img_array = np.array(img1)
    # print(img_array.shape)
    distance = ndi.distance_transform_edt(img_array)
    markers = []
    local_maxi = peak_local_max(distance, threshold_abs=9, indices=False, footprint=np.ones((1, 1)),
                                labels=img_array)
    markers = ndi.label(local_maxi)[0]
    ws_labels = watershed(-distance, markers, mask=img_array)

    return ws_labels

def WaterShedOptimizeK(img):
    img1 = img.convert("L")
    img_array = np.array(img1)
    # print(img_array.shape)
    distance = ndi.distance_transform_edt(img_array)
    markers = []
    local_maxi = peak_local_max(distance, min_distance=40,indices=False, labels=img_array)
    markers = ndi.label(local_maxi)[0]
    ws_labels = watershed(-distance, markers, mask=img_array)

    return ws_labels

for img_path in images:
    img = Image.open(img_path)
    img.thumbnail(size)  # Convert the image to 100 x 100
    # Convert the image to a numpy matrix
    
    ms_labels = MeanShifting(img)
    ws_labels = WaterShedding(img)

    # Display the results
    plot_three_images(img_path, img, "Original Image", ms_labels, "MeanShift Labels",
                      ws_labels, "Watershed Labels")


for img_path in ext_images:

    img = Image.open(img_path)
    img.thumbnail(size)

    # TODO: perform meanshift on image
    ms_labels = MeanShifting(img)  # CHANGE THIS

    # TODO: perform an optimisation and then watershed on image
    if img_path == 'kiwi.png':
        ws_labels = WaterShedOptimizeK(img)  # CHANGE THIS
    else:
        ws_labels = WaterShedOptimizeC(img)

    # Display the results
    plot_three_images(img_path, img, "Original Image", ms_labels, "MeanShift Labels",
                      ws_labels, "Watershed Labels")


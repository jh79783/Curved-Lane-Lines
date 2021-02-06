import cv2
import os.path as op
from glob import glob
import numpy as np


def main():
    files = glob("C:\\Users\\user\\Downloads\\3\\3\\*.jpg")
    for file in files:
        image = load_image(file)
        detect_lane(image)
        cv2.waitKey()


def load_image(file):
    image = cv2.imread(file)
    image = cv2.resize(image, (960, 600))
    # cv2.imshow("image", image)
    cropimg = image[310:550, 80:-80]
    print("read image:", op.basename(file), image.shape, cropimg.shape)
    cv2.imshow("crop", cropimg)
    return cropimg


def detect_lane(image):
    # color_edge(image)
    binary_image = region_binary(image)
    cv2.imshow("region binary", binary_image)


def color_edge(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    cv2.imshow("hls", hls)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    l_sobel = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)
    l_sobel_abs = np.absolute(l_sobel)
    l_sobel = np.uint8(255 * l_sobel_abs / np.max(l_sobel_abs))
    l_sobel = cv2.blur(l_sobel, (3,3))
    l_sobel_bin = np.zeros_like(l_sobel)
    l_sobel_bin[(l_sobel >= 15)] = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 100)] = 255

    white_lane_binary = np.bitwise_or(l_sobel_bin, s_binary)
    cv2.imshow("edge", white_lane_binary)


def color_binary(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    cv2.imshow("hls", hls)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    h_binary = np.zeros_like(h_channel)
    h_binary[l_channel < 40] = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[l_channel > 100] = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[s_channel < 100] = 255

    white_lane_binary = np.bitwise_and(np.bitwise_and(l_binary, s_binary), s_binary)
    cv2.imshow("white_lane", white_lane_binary)

    adaptive = cv2.adaptiveThreshold(l_binary, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)
    cv2.imshow("adaptive", adaptive)


def region_binary(image):
    rw, rh = 80, 40
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = np.zeros(image.shape[:2], np.uint8)
    for rx in range(0, image.shape[1], rw):
        for ry in range(0, image.shape[0], rh):
            region = gray[ry:ry+rh, rx:rx+rw]
            thresh = np.median(region).astype(int)
            # thresh, region_bin = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            thresh = min(thresh, 100)
            ret, region_bin = cv2.threshold(region, thresh+15, 255, cv2.THRESH_BINARY)
            # print("regionshape", region.shape, region_bin.shape, thresh)
            binary[ry:ry+rh, rx:rx+rw] = region_bin
            # print("thresh:", rx, ry, thresh)


    return binary


if __name__ == '__main__':
    main()

import numpy as np
import cv2
import glob
import os

import config


class LaneDetect:
    def __init__(self):
        self.src = None
        self.dst = None
        self.img_size = None

    def read_video(self):
        cap = cv2.VideoCapture(config.FILE)
        if not cap.isOpened():
            print("file error")

        while cap.isOpened():
            _, original_frame = cap.read()
            self.img_size = original_frame.shape[1::-1]
            hls_frame = self.hls_channel(original_frame)
            combined_binary = self.gradient_threshold(hls_frame)

            self.src = np.float32(
                [[(self.img_size[0] / 2) - 200, self.img_size[1] / 2 - 80],   #좌측 상단
                 [((self.img_size[0] / 6) + 50), self.img_size[1] - 200],     #좌측 하단
                 [(self.img_size[0] * 5 / 6) - 130, self.img_size[1] - 200],  #우측 하단
                 [(self.img_size[0] / 2 + 120), self.img_size[1] / 2 - 80]])  #우측 상단
            self.dst = np.float32(
                [[(self.img_size[0] / 4), 0],
                 [(self.img_size[0] / 4), self.img_size[1]],
                 [(self.img_size[0] * 3 / 4), self.img_size[1]],
                 [(self.img_size[0] * 3 / 4), 0]])

            bird_view = self.perspective(combined_binary, dst_size=self.img_size, src=self.src, dst=self.dst)
            bird_view_lane, left_fitx, right_fitx = self.fit_polynomial(bird_view)
            color_img = self.draw_lanes(original_frame, left_fitx, right_fitx)

            cv2.imshow("test", color_img)
            k = cv2.waitKey(33)
            if k == ord('q'):
                cap.release()
                break
        cv2.destroyAllWindows()

    def hls_channel(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        return hls

    def gradient_threshold(self, img, sobel_kernel=3, gradient_thresh=(15, 255)):
        h_channel = img[:, :, 0]
        l_channel = img[:, :, 1]
        s_channel = img[:, :, 2]
        sobelxy = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1, ksize=sobel_kernel)
        abs_sobel = np.absolute(sobelxy)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        gradient_binary = np.zeros_like(scaled_sobel)
        gradient_binary[(scaled_sobel >= gradient_thresh[0]) & (scaled_sobel <= gradient_thresh[1])] = 255
        color_binary = self.color_threshold(s_channel)

        combined_binary = np.zeros_like(gradient_binary)
        combined_binary[(color_binary == 255) | (gradient_binary == 255)] = 255
        return combined_binary

    def color_threshold(self, s_channel, color_thresh=(150, 255)):
        color_binary = np.zeros_like(s_channel)
        color_binary[(s_channel >= color_thresh[0]) & (s_channel <= color_thresh[1])] = 255
        return color_binary

    def perspective(self, img, dst_size=(1280, 720), src=None, dst=None):
        if src is not None:
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(img, M, dst_size, flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)
            return warped
        else:
            assert print("check src & dst value")

    def fit_polynomial(self, img):
        leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img = self.find_line(img, draw_window=True)

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            print("failed to fit a line")
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        out_img[lefty, leftx] = [255, 0, 100]
        out_img[righty, rightx] = [0, 100, 255]
        return out_img, left_fitx, right_fitx

    def find_line(self, img, nwindow=9, margin=100, draw_window=False):
        out_img = np.dstack((img, img, img))

        histogram = self.get_histogram(img)
        midpoint = int(histogram.shape[0]/2)
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint

        # window 높이 설정
        window_height = np.int(img.shape[0]/nwindow)

        nonzero = img.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])

        left_current = left_base
        right_current = right_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindow):
            window_y_low = img.shape[0] - (window+1)*window_height
            window_y_high = img.shape[0] - window*window_height
            window_left_low = left_current - margin
            window_left_high = left_current + margin
            window_right_low = right_current - margin
            window_right_high = right_current + margin

            if draw_window == True:
                cv2.rectangle(out_img, (window_left_low, window_y_low), (window_left_high, window_y_high), (100, 255,
                                                                                                            255), 3)
                cv2.rectangle(out_img, (window_right_low, window_y_low), (window_right_high, window_y_high), (100, 255,
                                                                                                              255), 3)
            good_left_inds = ((nonzeroy >= window_y_low) & (nonzeroy < window_y_high) &
                              (nonzerox >= window_left_low) & (nonzerox < window_left_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= window_y_low) & (nonzeroy < window_y_high) &
                               (nonzerox >= window_right_low) & (nonzerox < window_right_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > config.minpixel:
                left_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > config.minpixel:
                right_current = np.int(np.mean(nonzerox[good_right_inds]))

        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            pass

        # 왼쪽 오른쪽 선 픽셀위치 추출
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img

    def get_histogram(self, img):
        hist = np.sum(img[img.shape[0] // 2:, :], axis=0)
        return hist

    def draw_lanes(self, img, left_fitx, right_fitx):
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        color_img = np.zeros_like(img)

        left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        points = np.hstack((left, right))

        cv2.fillPoly(color_img, np.int_(points), (0, 200, 255))
        inv_perspective_img = self.inv_perspective(color_img, dst_size=self.img_size, src=self.dst, dst=self.src)
        inv_perspective_img = cv2.addWeighted(img, 1, inv_perspective_img, 0.7, 0)
        return inv_perspective_img

    def inv_perspective(self, img, dst_size=(1280, 720), src=None, dst=None):
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, dst_size)
        return warped


def main():
    LaneDetect().read_video()


if __name__ == "__main__":
    main()


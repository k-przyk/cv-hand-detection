#!/usr/bin python3.6

import rclpy
from rclpy.node import Node

import json
import cv2
import numpy as np


class HSVMask(Node):
    def __init__(self):
        super().__init__("HSV_Mask")

        # Timer for ros node callback
        exec_timer = 0.01  # In seconds
        self.timer = self.create_timer(exec_timer, self.create_mask)

        # Windows for frames and frame capture
        self.video = cv2.VideoCapture(0)
        cv2.namedWindow("mask")
        cv2.namedWindow("camera")

        # Create Trackbars for HSV Thresholding
        self.trackbars = [
            "High Hue",
            "Low Hue",
            "High Sat",
            "Low Sat",
            "High Val",
            "Low Val",
        ]
        for bar in self.trackbars:
            cv2.createTrackbar(bar, "mask", 0, 255, self.nothing)

        self.values = np.arange(len(self.trackbars))

    def create_mask(self):

        ret, image = self.video.read()
        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        for bar, i in zip(self.trackbars, range(len(self.values))):
            self.values[i] = cv2.getTrackbarPos(bar, "mask")

        threshold = cv2.inRange(
            hsv_frame, np.array(self.values[1::2]), np.array(self.values[0::2])
        )
        hsv_mask = cv2.bitwise_and(image, image, mask=threshold)

        cv2.imshow("camera", image)
        cv2.imshow("mask", hsv_mask)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            hsv_mask_values = json.dumps(self.values.tolist())

            from pathlib import Path, PurePath

            path_to_parent = Path(__file__).parent.parent.absolute()
            path_to_mask = path_to_parent / "masks" / "hsv_mask_values.json"

            print(path_to_mask)

            try:
                f = open(path_to_mask, "x")
                f.write(hsv_mask_values)
                f.close()

            except FileExistsError:
                input_msg = (
                    "There is an existing hsv_mask_values.json. "
                    + "Would you like to overwrite it? (y/n) \n"
                )
                write = input(input_msg)
                if write == "y":

                    path_to_mask.unlink()
                    # os.remove("hsv_mask_values.json")
                    f = open(path_to_mask, "x")
                    f.write(hsv_mask_values)
                    f.close()

            self.destroy_node()
            exit()

    def nothing(self, x):  # print(self.values)
        pass


def main(args=None):
    rclpy.init(args=args)

    hsv_mask = HSVMask()
    rclpy.spin(hsv_mask)

    # Destroy the node upon task end
    cv2.destroyAllWindows()
    hsv_mask.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

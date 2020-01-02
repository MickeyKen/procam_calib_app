import argparse
import cv2
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="This script creates a circleboard image for calibration")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument("--margin_size", type=int, default=12)
    parser.add_argument("--block_size", type=int, default=100)
    parser.add_argument("--radius", type=int, default=20)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    w = args.width
    h = args.height
    margin = args.margin_size
    block_size = args.block_size
    radius = args.radius
    chessboard = np.ones((block_size * h + margin * 2, block_size * w + margin * 2), dtype=np.uint8) * 255

    for y in range(h):
        for x in range(w):
            cx = int((x + 0.5) * block_size + margin)
            cy = int((y + 0.5) * block_size + margin)
            cv2.circle(chessboard, (cx, cy), radius, 0, thickness=-1)


    ch = 1920
    cw = 1080
    chessboard = cv2.resize(chessboard,(ch,cw))

    cv2.imwrite("circleboard{}x{}.png".format(w, h), chessboard)
    cv2.imshow("circleboard", chessboard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

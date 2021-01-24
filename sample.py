import copy
import argparse

import cv2 as cv

from color_equalize_hist import color_equalize_hist


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_CLAHE', action='store_true')

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    use_CLAHE = args.use_CLAHE
    device = args.device
    cap_width = args.width
    cap_height = args.height

    cap = cv.VideoCapture(device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        original_frame = copy.deepcopy(frame)

        frame = color_equalize_hist(frame, use_CLAHE)

        cv.imshow('original', original_frame)
        cv.imshow('polygon filter', frame)
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

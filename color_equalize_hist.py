import copy
import cv2 as cv


def color_equalize_hist(bgr_image,
                        use_CLAHE=False,
                        clipLimit=2.0,
                        tileGridSize=(8, 8)):
    """ヒストグラム平坦化(カラー画像)した画像を返す

    Args:
        bgr_image: OpenCV Image
        use_CLAHE: use CLAHE
        clipLimit: CLAHE Option
        tileGridSize: CLAHE Option

    Returns:
        Image after applying the color_equalize_hist.
    """
    _bgr_image = copy.deepcopy(bgr_image)

    yuv_image = cv.cvtColor(_bgr_image, cv.COLOR_BGR2YUV)

    if not use_CLAHE:
        yuv_image[:, :, 0] = cv.equalizeHist(yuv_image[:, :, 0])
    else:
        clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        yuv_image[:, :, 0] = clahe.apply(yuv_image[:, :, 0])

    result_image = cv.cvtColor(yuv_image, cv.COLOR_YUV2BGR)

    return result_image


if __name__ == '__main__':
    sample_image = cv.imread('sample.jpg')
    result_image = color_equalize_hist(sample_image, True)

    cv.imshow('Before', sample_image)
    cv.imshow('After', result_image)
    cv.waitKey(-1)

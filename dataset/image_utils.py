import numpy as np

def extract_random_roi(img_shape: np.ndarray, min_percentage: float, max_percentage: float, ratio: float,
                       padding: int = 0) -> (float, float, float, float):
    """
    :param img_shape: shape of the image to take the roi from, shape: HxW
    :param min_percentage: the minimum size of the roi for both width and height
    :param max_percentage: the maximum size of the roi for both width and height
    :param ratio: width/height ratio
    :padding: padding to put the image away from the borders
    :return: x, y, w, h of the roi
    """
    assert min_percentage < max_percentage, " min_percentage should be less than max_percentage"
    assert len(img_shape) == 2, "Image should be in shape HxW"
    h_min = img_shape[0] * min_percentage
    h_max = img_shape[0] * max_percentage

    h = np.random.randint(h_min, h_max)
    w = np.int32((1 / ratio) * h)

    y_min, x_min = padding, padding
    y_max = img_shape[0] - h - padding
    x_max = img_shape[1] - w - padding

    y = np.random.randint(y_min, y_max)
    x = np.random.randint(x_min, x_max)

    return int(x), int(y), int(w), int(h)

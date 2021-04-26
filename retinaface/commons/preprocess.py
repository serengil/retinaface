import numpy as np
import cv2

#this function is copied from the following code snippet: https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/retinaface.py
def resize_image(img, scales):
    img_w, img_h = img.shape[0:2]
    target_size = scales[0]
    max_size = scales[1]

    if img_w > img_h:
        im_size_min, im_size_max = img_h, img_w
    else:
        im_size_min, im_size_max = img_w, img_h

    im_scale = target_size / float(im_size_min)

    if np.round(im_scale * im_size_max) > max_size:
        im_scale = max_size / float(im_size_max)

    if im_scale != 1.0:
        img = cv2.resize(
            img,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR
        )

    return img, im_scale

#this function is copied from the following code snippet: https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/retinaface.py
def preprocess_image(img):
    pixel_means = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pixel_stds = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    pixel_scale = float(1.0)
    scales = [1024, 1980]

    img, im_scale = resize_image(img, scales)
    img = img.astype(np.float32)
    im_tensor = np.zeros((1, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

    # Make image scaling + BGR2RGB conversion + transpose (N,H,W,C) to (N,C,H,W)
    for i in range(3):
        im_tensor[0, :, :, i] = (img[:, :, 2 - i] / pixel_scale - pixel_means[2 - i]) / pixel_stds[2 - i]

    return im_tensor, img.shape[0:2], im_scale

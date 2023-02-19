import numpy as np
import cv2


# This function is modified from the following code snippet:
# https://github.com/StanislasBertrand/RetinaFace-tf2/blob/5f68ce8130889384cb8aca937a270cea4ef2d020/retinaface.py#L49-L74
def resize_image(img, scales, allow_upscaling):
    img_h, img_w = img.shape[0:2]
    target_size = scales[0]
    max_size = scales[1]

    if img_w > img_h:
        im_size_min, im_size_max = img_h, img_w
    else:
        im_size_min, im_size_max = img_w, img_h

    im_scale = target_size / float(im_size_min)
    if not allow_upscaling:
        im_scale = min(1.0, im_scale)

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

def preprocess_batch_images(batch_images):
    images = []
    img_scales = []
    scales = [1980, 1024]
    for image in batch_images:
        img_h, img_w = image.shape[0:2]
        scale_x = scales[0] / img_w
        scale_y = scales[1] / img_h
        scale = scale_x
        if scale_y<scale_x<=1 or 1<=scale_y<scale_x or scale_y<=1<=scale_x:
            scale = scale_y

        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        if scale == scale_x:
            space = scales[1] - image.shape[0]
            image = np.concatenate((image, np.zeros((space, scales[0], 3), np.uint8)), axis=0)
        elif scale == scale_y:
            space = scales[0] - image.shape[1]
            image = np.concatenate((image, np.zeros((scales[1], space, 3), np.uint8)), axis=1)

        images.append(image.astype(np.float32))
        img_scales.append(scale)

    return np.asarray(images), img_scales

# This function is modified from the following code snippet:
# https://github.com/StanislasBertrand/RetinaFace-tf2/blob/5f68ce8130889384cb8aca937a270cea4ef2d020/retinaface.py#L76-L96
def preprocess_image(img, allow_upscaling):
    pixel_means = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pixel_stds = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    pixel_scale = float(1.0)
    scales = [1024, 1980]

    img, im_scale = resize_image(img, scales, allow_upscaling)
    img = img.astype(np.float32)
    im_tensor = np.zeros((1, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

    # Make image scaling + BGR2RGB conversion + transpose (N,H,W,C) to (N,C,H,W)
    for i in range(3):
        im_tensor[0, :, :, i] = (img[:, :, 2 - i] / pixel_scale - pixel_means[2 - i]) / pixel_stds[2 - i]

    return im_tensor, img.shape[0:2], im_scale

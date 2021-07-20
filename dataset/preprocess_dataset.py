import tensorflow as tf
import numpy as np
import sys
import os
import cv2

sys.path.append('/project/ai/scratch01/salyua01/sharing/guide_acc')
img_dir='./imagenet-sample-images/'
label_file='./imagelabel.txt'

#MNASNET PARAM
input_height=224
input_width=224
input_channel = 3
mean = 127.5
var = 127.5

tf.enable_eager_execution()

def smallest_size_at_least(height, width, resize_min):
    """Computes new shape with the smallest side equal to `smallest_side`.

    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.

    Args:
      height: an int32 scalar tensor indicating the current height.
      width: an int32 scalar tensor indicating the current width.
      resize_min: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      new_height: an int32 scalar tensor indicating the new height.
      new_width: an int32 scalar tensor indicating the new width.
    """
    resize_min = tf.cast(resize_min, tf.float32)

    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim

    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(tf.round(height * scale_ratio), tf.int32)
    new_width = tf.cast(tf.round(width * scale_ratio), tf.int32)

    return new_height, new_width

def resize_image(image, height, width, method='BILINEAR'):
    """Simple wrapper around tf.resize_images.

    This is primarily to make sure we use the same `ResizeMethod` and other
    details each time.

    Args:
      image: A 3-D image `Tensor`.
      height: The target height for the resized image.
      width: The target width for the resized image.

    Returns:
      resized_image: A 3-D tensor containing the resized image. The first two
        dimensions have the shape [height, width].
    """
    resize_func = tf.image.ResizeMethod.NEAREST_NEIGHBOR if method == 'NEAREST' else tf.image.ResizeMethod.BILINEAR
    return tf.image.resize_images(image, [height, width], method=resize_func, align_corners=False)


def aspect_preserving_resize(image, resize_min, channels=3, method='BILINEAR'):
    """Resize images preserving the original aspect ratio.

    Args:
      image: A 3-D image `Tensor`.
      resize_min: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      resized_image: A 3-D tensor containing the resized image.
    """
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    new_height, new_width = smallest_size_at_least(height, width, resize_min)
    return resize_image(image, new_height, new_width, method)


def central_crop(image, crop_height, crop_width, channels=3):
    """Performs central crops of the given image list.

    Args:
      image: a 3-D image tensor
      crop_height: the height of the image following the crop.
      crop_width: the width of the image following the crop.

    Returns:
      3-D tensor with cropped image.
    """
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    amount_to_be_cropped_h = height - crop_height
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = width - crop_width
    crop_left = amount_to_be_cropped_w // 2
    # return tf.image.crop_to_bounding_box(image, crop_top, crop_left, crop_height, crop_width)

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(height, crop_height),
            tf.greater_equal(width, crop_width)),
        ['Crop size greater than the image size.']
    )
    with tf.control_dependencies([size_assertion]):
        if channels == 1:
            image = tf.squeeze(image)
            crop_start = [crop_top, crop_left, ]
            crop_shape = [crop_height, crop_width, ]
        elif channels >= 3:
            crop_start = [crop_top, crop_left, 0]
            crop_shape = [crop_height, crop_width, -1]

        image = tf.slice(image, crop_start, crop_shape)

    return tf.reshape(image, [crop_height, crop_width, -1])

label_data = open(label_file)
filename_list = []
label_list = []
for line in label_data:
    filename_list.append(line.rstrip('\n').split(' ')[0])
    label_list.append(int(line.rstrip('\n').split(' ')[1]))
label_data.close()
img_num = len(label_list)

images = np.zeros([img_num, input_height, input_width, input_channel], np.float32)
for file_name, img_idx in zip(filename_list, range(img_num)):   
    image_file = os.path.join(img_dir, file_name)
    img_s = tf.gfile.GFile(image_file, 'rb').read()  
    image = tf.image.decode_jpeg(img_s)
    image = tf.cast(image, tf.float32)
    image = tf.clip_by_value(image, 0., 255.)
    image = aspect_preserving_resize(image, min(input_height, input_width), input_channel) 
    image = central_crop(image, input_height, input_width)
    image = tf.image.resize_images(image, [input_height, input_width])
    image = (image - mean) / var
    image = image.numpy()
    _, _, ch = image.shape
    if ch == 1:
        image = tf.tile(image, multiples=[1,1,3])
        image = image.numpy()
    images[img_idx] = image

np.save('dataset.npy', images)

labels = np.array(label_list)
np.save('label.npy', labels)
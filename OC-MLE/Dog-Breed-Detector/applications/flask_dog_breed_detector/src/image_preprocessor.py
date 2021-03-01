import io
import os
import tensorflow as tf
# For downloading the image.
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import cv2
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def display_image(image):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def download_and_resize_image(url, new_width=256, new_height=256, display=False):
    _, filename = tempfile.mkstemp(suffix=".jpg")
    response = urlopen(url)
    image_data = response.read()
    image_data = BytesIO(image_data)
    pil_image = Image.open(image_data)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    print("Image downloaded to %s." % filename)
    if display:
        display_image(pil_image)
    return filename


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()):
    """
    Adds a bounding box to an image.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, selected_classes=None, max_boxes=10, min_score=0.1):
    """
    Overlay labeled boxes on an image with formatted scores and label names.
    """
    colors = list(ImageColor.colormap.values())
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            decoded_class_name = class_names[i].decode("ascii").lower()
            display_str = "{}: {}%".format(decoded_class_name, int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            selected_classes_cond = ((selected_classes is not None) and (decoded_class_name in selected_classes))
            if selected_classes_cond or (selected_classes is None):
                draw_bounding_box_on_image(image_pil,
                                           ymin,
                                           xmin,
                                           ymax,
                                           xmax,
                                           color,
                                           font,
                                           display_str_list=[display_str])
                np.copyto(image, np.array(image_pil))
    return image


def get_img_array(img_path, size):
    """
    Transform image file (from file path) into image array

    :param img_path: image file path (absolute or relative)
    :param size: image size (width, height)
    
    :return: image as array

    """
    # `img` is a PIL image of size w x h
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (w, h, n_ch)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, w, h, n_ch)
    array = np.expand_dims(array, axis=0)
    return array


# Grad CAM
# Paper : https://arxiv.org/pdf/1610.02391.pdf
# Article : https://medium.com/@mohamedchetoui/grad-cam-gradient-weighted-class-activation-mapping-ffd72742243a


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, base_model=None):
    """

    Build Grad-CAM heatmap from image array and model data

    :param img_array: an image numpy array
    :param model: a TensorFlow classifier
    :param last_conv_layer_name: last convolutional output layer name
    :param base_model: base model from custom Sequential model (optional)
    
    :return: a heatmap (array)
    
    """
    # First, we create a model that maps the input image to the activations of the last conv layer
    if base_model is None:
        base_model = model
        last_conv_layer_idx = list(map(lambda l: l.name, base_model.layers)).index(last_conv_layer_name)
        classifier_layer_names = [layer.name for i, layer in enumerate(base_model.layers) if i > last_conv_layer_idx]
    else:
        classifier_layer_names = [layer.name for layer in model.layers if layer.name != base_model.name]
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(base_model.inputs, last_conv_layer.output)
    # Second, we create a model that maps the activations of the last conv layer to the final class predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    # This is the gradient of the top predicted class with regard to the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    # This is a vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    # The channel-wise mean of the resulting feature map is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def grad_cam(img_path, *args):
    """
    Generate Grad-CAM PIL image
    
    :param img_path: relative path to image file
    :param *args: make_gradcam_heatmap parameters

    :return: a PIL image

    """
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(*args)
    # We load the original image
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    # We use jet colormap to colorize heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # Superimpose the heatmap on original image
    superimposed_img = np.float32(heatmap) + np.float32(img)
    superimposed_img = 255 * superimposed_img / np.max(superimposed_img)
    superimposed_img = superimposed_img[:, :, ::-1]
    superimposed_img = Image.fromarray(np.uint8(superimposed_img))
    return superimposed_img

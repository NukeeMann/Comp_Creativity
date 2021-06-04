import tkinter as tk
import os
import ntpath
import tensorflow as tf
import matplotlib as mpl

style_predict_path = tf.keras.utils.get_file('style_predict.tflite',
                                             'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite')
style_transform_path = tf.keras.utils.get_file('style_transform.tflite',
                                               'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite')
destination_folder = os.path.join('images','stylized_images')


# Function to load an image from a file, and add a batch dimension.
def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]

    return img


# Function to pre-process by resizing an central cropping it.
def preprocess_image(image, target_dim):
    # Resize the image so that the shorter dimension becomes 256px.
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)

    # Central crop the image.
    image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

    return image


def imsave(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    tf.keras.preprocessing.image.save_img(title, image)


# Function to run style prediction on preprocessed style image.
def run_style_predict(preprocessed_style_image):
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=style_predict_path)

    # Set model input.
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

    # Calculate style bottleneck.
    interpreter.invoke()
    style_bottleneck = interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
    )()

    return style_bottleneck


# Run style transform on preprocessed style image
def run_style_transform(style_bottleneck, preprocessed_content_image):
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=style_transform_path)

    # Set model input.
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()

    # Set model inputs.
    interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
    interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
    interpreter.invoke()

    # Transform content image.
    stylized_image = interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
    )()

    return stylized_image


def transfer_style(image_path, style_path):
    mpl.rcParams['figure.figsize'] = (12, 12)
    mpl.rcParams['axes.grid'] = False
    style_image = load_img(style_path)
    preprocessed_style_image = preprocess_image(style_image, 256)
    content_image = load_img(image_path)
    preprocessed_content_image = preprocess_image(content_image, 384)
    style_bottleneck = run_style_predict(preprocessed_style_image)
    stylized_image = run_style_transform(style_bottleneck, preprocessed_content_image)
    path = os.path.join(destination_folder, 'stylized_' + ntpath.basename(image_path) + '.jpg')
    imsave(stylized_image, path)
    return path

from tkinter import filedialog
from PIL import Image, ImageTk

class styleTransfer(tk.Frame):
    def browseImage(self):
        self.image_file = filedialog.askopenfilename(initialdir="/",
                                                title="Select a File",
                                                filetypes=(("jpeg files",
                                                            "*.jpeg"),
                                                           ("all files",
                                                            "*.*")))
        tk.Label(self, text=self.image_file).grid(row=1, column=0)

    def browseStyle(self):
        self.style_file = filedialog.askopenfilename(initialdir="/",
                                                title="Select a File",
                                                filetypes=(("jpg files",
                                                            "*.jpg"),
                                                           ("all files",
                                                            "*.*")))
        tk.Label(self, text=self.style_file).grid(row=3, column=0)
    def transfer(self):
        if self.image_file != '' and self.style_file != '':
            img = transfer_style(self.image_file, self.style_file)
            image = Image.open(img)
            test = ImageTk.PhotoImage(image)
            label1 = tk.Label(self, image=test)
            label1.image = test
            label1.grid(row=5, column=0)

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.image_file = ''
        self.style_file = ''
        tk.Button(self, text='Browse content image', command=self.browseImage).grid(row=0, column=0)
        tk.Label(self, text='').grid(row=1, column=0)
        tk.Button(self, text='Browse style image', command=self.browseStyle).grid(row=2, column=0)
        tk.Label(self, text='').grid(row=3, column=0)
        tk.Button(self, text='Transfer style', command=self.transfer).grid(row=4, column=0)


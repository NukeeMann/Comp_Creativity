import tkinter as tk
import os
import tensorflow as tf
from tkinter import filedialog
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageTk
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


class AST(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.image_file = ''
        self.style_file = ''
        self.button1 = tk.Button(self, text='Browse content image', command=self.browse_image).grid(row=0, column=0)
        self.label1 = tk.Label(self, text='').grid(row=1, column=0)
        self.button2 = tk.Button(self, text='Browse style image', command=self.browse_style).grid(row=2, column=0)
        self.label2 = tk.Label(self, text='').grid(row=3, column=0)
        self.button3 = tk.Button(self, text='Transfer style', command=self.transform).grid(row=4, column=0)
        self.choices = [256, 512, 1024, 2048, 4096]
        self.max_dim = tk.IntVar(self)
        self.max_dim.set(self.choices[0])
        self.choosebox = tk.OptionMenu(self, self.max_dim, *self.choices).grid(row=5, column=0)


    def browse_image(self):
        self.image_file = filedialog.askopenfilename(initialdir="/",
                                                     title="Select a File",
                                                     filetypes=(("jpeg files", "*.jpg"),
                                                                ("gif files", "*.gif*"),
                                                                ("png files", "*.png"),
                                                                ("all files", "*.*")))
        tk.Label(self, text=self.image_file).grid(row=1, column=0)

    def browse_style(self):
        self.style_file = filedialog.askopenfilename(initialdir="/",
                                                     title="Select a File",
                                                     filetypes=(("jpeg files", "*.jpg"),
                                                                ("gif files", "*.gif*"),
                                                                ("png files", "*.png"),
                                                                ("all files", "*.*")))
        tk.Label(self, text=self.style_file).grid(row=3, column=0)


    def transform(self):
        if self.image_file != '' and self.style_file != '':
            hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
            content_image = self.load_img(self.image_file)
            style_image = self.load_img(self.style_file, self.max_dim.get())
            stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
            output_img = self.tensor_to_image(stylized_image)

            output_img = ImageTk.PhotoImage(output_img)
            label = tk.Label(self, image=output_img)
            label.image = output_img
            label.grid(row=7, column=0)

    @staticmethod
    def load_img(path_to_img, max_dim=512):
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    @staticmethod
    def tensor_to_image(tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)

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
        tk.Frame.__init__(self, parent, highlightbackground="blue", highlightthickness=1)
        self.image_file = ''
        self.style_file = ''
        top_padding = 400
        self.choices = [256, 512, 1024, 2048, 4096]
        self.max_dim = tk.IntVar(self)
        self.max_dim.set(self.choices[0])
        self.preview_image = tk.Label(self, text='PREVIEW', font=("TkDefaultFont", 80), fg='white',
                                      background="black").place(
                                        x=140, y=20, height=360, width=640)
        self.label1 = tk.Label(self, text='LABRADORy.png', font=44, background="lightgrey").place(
                                        x=60, y=top_padding + 20, height=40, width=400)
        self.button1 = tk.Button(self, text='Browse content image', font=44, command=self.browse_image).place(
                                        x=460, y=top_padding + 20, height=40, width=400)

        self.label2 = tk.Label(self, text='wan_gog.png', font=44, background="lightgrey").place(
                                        x=60, y=top_padding + 61, height=40, width=400)
        self.button2 = tk.Button(self, text='Browse style image', font=44, command=self.browse_style).place(
                                        x=460, y=top_padding + 61, height=40, width=400)

        self.choose_box = tk.OptionMenu(self, self.max_dim, *self.choices)
        self.choose_box.config(font=44)
        dropdown = self.nametowidget(self.choose_box.menuname).config(font=44)
        self.choose_box.place(x=60, y=top_padding + 102, height=40, width=400)

        self.button3 = tk.Button(self, text='Transfer style', font=44, command=self.transform).place(
                                        x=460, y=top_padding + 102, height=40, width=400)

    # Browse image to transform
    def browse_image(self):
        self.image_file = filedialog.askopenfilename(initialdir="/",
                                                     title="Select a File",
                                                     filetypes=(("jpeg files", "*.jpg"),
                                                                ("gif files", "*.gif*"),
                                                                ("png files", "*.png"),
                                                                ("all files", "*.*")))
        tk.Label(self, text=self.image_file).grid(row=1, column=0)

    # Browse image to get style from
    def browse_style(self):
        self.style_file = filedialog.askopenfilename(initialdir="/",
                                                     title="Select a File",
                                                     filetypes=(("jpeg files", "*.jpg"),
                                                                ("gif files", "*.gif*"),
                                                                ("png files", "*.png"),
                                                                ("all files", "*.*")))
        tk.Label(self, text=self.style_file).grid(row=3, column=0)

    # Transform image
    def transform(self):
        if self.image_file != '' and self.style_file != '':
            tf.compat.v1.enable_eager_execution()
            # Load the model
            hub_model = hub.load('algorithms/models/ATS')
            # Load images
            content_image = self.load_img(self.image_file)
            style_image = self.load_img(self.style_file, self.max_dim.get())
            # Transform image
            stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
            output_img = self.tensor_to_image(stylized_image)

            # Show transformed image
            output_img = ImageTk.PhotoImage(output_img)
            label = tk.Label(self, image=output_img)
            label.image = output_img
            label.grid(row=7, column=0)

    @staticmethod
    def load_img(path_to_img, max_dim=512):
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
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

import tkinter as tk
from tkinter import filedialog
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import moviepy.editor as mpe
import numpy as np
from scipy.stats import truncnorm
import tensorflow_hub as hub
from tkinter.messagebox import showerror
from threading import Thread
from concurrent.futures import Future
import algorithms.MorphingLabels as MorphingLabels
import tarfile, requests
import os
from PIL import ImageTk

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

def call_with_future(fn, future, args, kwargs):
    try:
        result = fn(*args, **kwargs)
        future.set_result(result)
    except Exception as exc:
        future.set_exception(exc)

def threaded(fn):
    def wrapper(*args, **kwargs):
        future = Future()
        Thread(target=call_with_future, args=(fn, future, args, kwargs)).start()
        return future
    return wrapper

class Morphing(tk.Frame):
    num_samples = 1
    num_interps = 100
    truncation = 0.2
    noise_seed_A = 0
    noise_seed_B = 0

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        top_padding = 50
        self.video = None
        self.label_select_styl = tk.Label(self, text='Choose images to create morphing: ', font=("TkDefaultFont", 16))
        self.label_select_styl.place(x=60, y=top_padding-10, height=30, width=350)
        self.choices = MorphingLabels.get_labels()
        self.image_number_1 = tk.StringVar(self)
        self.image_number_1.set(self.choices[0])

        self.choose_box_param1 = tk.OptionMenu(self, self.image_number_1, *self.choices, command=self.option_changed)
        self.choose_box_param1.config(font=("TkDefaultFont", 12))
        dropdown1 = self.nametowidget(self.choose_box_param1.menuname).config(font=("TkDefaultFont", 12))
        self.choose_box_param1.place(x=460, y=top_padding + 20, height=40, width=400)

        self.image_number_2 = tk.StringVar(self)
        self.image_number_2.set(self.choices[0])
        self.choose_box_param2 = tk.OptionMenu(self, self.image_number_2, *self.choices, command=self.option_changed)
        self.choose_box_param2.config(font=("TkDefaultFont", 12))
        dropdown2 = self.nametowidget(self.choose_box_param2.menuname).config(font=("TkDefaultFont", 12))
        self.choose_box_param2.place(x=460, y=top_padding + 61, height=40, width=400)

        self.label1 = tk.Label(self, text='Choose first image:', font=44, background="lightgrey").place(
                                        x=60, y=top_padding + 20, height=40, width=400)
        self.label2 = tk.Label(self, text='Choose second image:', font=44, background="lightgrey").place(
                                        x=60, y=top_padding + 61, height=40, width=400)
        self.generate_button = tk.Button(self, text='Generate morphing', font=44, command=self.create_morphing).place(
                                        x=60, y=top_padding + 102, height=40, width=400)
        self.save_button = tk.Button(self, text='Save result', font=44, bg='green', command=self.save_morphing).place(
                                        x=460, y=top_padding + 102, height=40, width=400)
        self.label1 = tk.Label(self, text='PREVIEW', font=("TkDefaultFont", 16)).place(x=410, y=220, height=40, width=100)
        self.preview_image1 = tk.Label(self, background="black")
        self.preview_image1.place(x=137, y=280, height=256, width=256)
        self.preview_image2 = tk.Label(self, background="black")
        self.preview_image2.place(x=537, y=280, height=256, width=256)

        self.option_changed()
        # Load the model
        self.load_model_h = self.loadModel()

    def option_changed(self, *args):
        image1_path = os.path.join("algorithms", "models", "morphing_imgs", MorphingLabels.get_img(self.image_number_1.get()))
        image2_path = os.path.join("algorithms", "models", "morphing_imgs", MorphingLabels.get_img(self.image_number_2.get()))
        img1 = ImageTk.PhotoImage(file=image1_path)
        img2 = ImageTk.PhotoImage(file=image2_path)
        self.preview_image1.config(image=img1)
        self.preview_image1.image = img1
        self.preview_image1.place(x=137, y=280, height=256, width=256)
        self.preview_image2.config(image=img2)
        self.preview_image2.image = img2
        self.preview_image2.place(x=537, y=280, height=256, width=256)

    @threaded
    def loadModel(self):
        tmp_path = os.path.join("algorithms", "models", "MORPHING")
        file_name = os.path.join("algorithms", "models", "MORPHING", "tmp.tar.gz")
        if not os.path.exists(tmp_path) or len(os.listdir(tmp_path)) == 0:
            if not os.path.exists(tmp_path):
                os.mkdir(tmp_path)
            url = 'https://tfhub.dev/deepmind/biggan-deep-128/1?tf-hub-format=compressed'
            print("Downloading BigGAN module")
            r = requests.get(url, allow_redirects=True)
            open(file_name, 'wb').write(r.content)
            file = tarfile.open(file_name)
            print("Extracting BigGAN module")
            file.extractall(tmp_path)
            file.close()
            os.remove(file_name)

        tf.reset_default_graph()
        tf.compat.v1.disable_eager_execution()
        print('Loading BigGAN module')
        self.module = hub.Module('algorithms/models/MORPHING')
        self.inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
                       for k, v in self.module.get_input_info_dict().items()}
        self.output = self.module(self.inputs)
        self.input_z = self.inputs['z']
        self.input_y = self.inputs['y']
        self.input_trunc = self.inputs['truncation']
        self.dim_z = self.input_z.shape.as_list()[1]
        self.vocab_size = self.input_y.shape.as_list()[1]
        initializer = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(initializer)
        print("Morphing is ready to go!!!")

    def save_morphing(self):
        if self.video is None:
            tk.messagebox.showerror(title="Error", message="There is nothing to save. Create morphing first.")
            return

        filename = filedialog.asksaveasfile(initialdir="results", mode='wb', defaultextension=".mp4",
                                            filetypes=(("MP4", "*.mp4"), ("all files", "*.*")))
        if not filename:
            return
        self.video.write_videofile(filename.name, fps=100)

    def truncated_z_sample(self, batch_size, truncation=1., seed=None):
        state = None if seed is None else np.random.RandomState(seed)
        values = truncnorm.rvs(-2, 2, size=(batch_size, self.dim_z), random_state=state)
        return truncation * values

    def one_hot(self, index, vocab_size):
        index = np.asarray(index)
        if len(index.shape) == 0:
            index = np.asarray([index])
        assert len(index.shape) == 1
        num = index.shape[0]
        output = np.zeros((num, vocab_size), dtype=np.float32)
        output[np.arange(num), index] = 1
        return output

    def one_hot_if_needed(self, label, vocab_size):
        label = np.asarray(label)
        if len(label.shape) <= 1:
            label = self.one_hot(label, vocab_size)
        assert len(label.shape) == 2
        return label

    # Function generating image from noise using GAN
    def sample(self, sess, noise, label, vocab_size, truncation=1., batch_size=8,):
        noise = np.asarray(noise)
        label = np.asarray(label)
        num = noise.shape[0]
        if len(label.shape) == 0:
            label = np.asarray([label] * num)
        if label.shape[0] != num:
            raise ValueError('Got # noise samples ({}) != # label samples ({})'
                             .format(noise.shape[0], label.shape[0]))
        label = self.one_hot_if_needed(label, vocab_size)
        ims = []
        for batch_start in range(0, num, batch_size):
            s = slice(batch_start, min(num, batch_start + batch_size))
            feed_dict = {self.input_z: noise[s], self.input_y: label[s], self.input_trunc: truncation}
            ims.append(sess.run(self.output, feed_dict=feed_dict))
        ims = np.concatenate(ims, axis=0)
        assert ims.shape[0] == num
        ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
        ims = np.uint8(ims)
        return ims

    # Basic interpolation function
    def interpolate(self, A, B, num_interps):
        if A.shape != B.shape:
            raise ValueError('A and B must have the same shape to interpolate.')
        alphas = np.linspace(0, 1, num_interps)
        return np.array([(1 - a) * A + a * B for a in alphas])

    # Function interpolating values from images (noises from which GAN generates images) to create mix of them
    def interpolate_and_shape(self, A, B, num_interps):
        interps = self.interpolate(A, B, num_interps)
        return (interps.transpose(1, 0, *range(2, len(interps.shape)))
                .reshape(self.num_samples * num_interps, *interps.shape[2:]))

    # Function creating video showing morphing between 2 photos generated by GAN
    def create_morphing(self):
        # Check if model is loaded
        self.load_model_h.result()
        category_A = MorphingLabels.get_value(self.image_number_1.get())
        category_B = MorphingLabels.get_value(self.image_number_2.get())
        z_A, z_B = [self.truncated_z_sample(self.num_samples, self.truncation, noise_seed)
                    for noise_seed in [self.noise_seed_A, self.noise_seed_B]]
        y_A, y_B = [self.one_hot([category] * self.num_samples, self.vocab_size)
                    for category in [category_A, category_B]]
        z_interp = self.interpolate_and_shape(z_A, z_B, self.num_interps)
        y_interp = self.interpolate_and_shape(y_A, y_B, self.num_interps)
        ims = self.sample(self.sess, z_interp, y_interp, self.vocab_size, truncation=self.truncation)
        video_name = 'video.avi'
        height, width, layers = ims[0].shape
        self.video = cv2.VideoWriter(video_name, 0, 50, (width, height))
        for img in ims:
            self.video.write(img)
        cv2.destroyAllWindows()
        self.video.release()
        self.video = mpe.VideoFileClip(video_name)
        os.startfile("video.avi")

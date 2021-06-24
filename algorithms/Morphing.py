import tkinter as tk
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import os
import moviepy.editor as mpe
import numpy as np
import PIL.Image
from scipy.stats import truncnorm
import tensorflow_hub as hub

class Morphing(tk.Frame):
    num_samples = 1
    num_interps = 100
    truncation = 0.2
    noise_seed_A = 0
    noise_seed_B = 0

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

    def interpolate(self, A, B, num_interps):
        if A.shape != B.shape:
            raise ValueError('A and B must have the same shape to interpolate.')
        alphas = np.linspace(0, 1, num_interps)
        return np.array([(1 - a) * A + a * B for a in alphas])

    def interpolate_and_shape(self, A, B, num_interps):
        interps = self.interpolate(A, B, num_interps)
        return (interps.transpose(1, 0, *range(2, len(interps.shape)))
                .reshape(self.num_samples * num_interps, *interps.shape[2:]))

    def create_morphing(self):
        self.module_path = 'https://tfhub.dev/deepmind/biggan-deep-256/1'
        tf.reset_default_graph()
        print('Loading BigGAN module from:', self.module_path)
        self.module = hub.Module(self.module_path)
        self.inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
                  for k, v in self.module.get_input_info_dict().items()}
        self.output = self.module(self.inputs)
        self.input_z = self.inputs['z']
        self.input_y = self.inputs['y']
        self.input_trunc = self.inputs['truncation']
        self.dim_z = self.input_z.shape.as_list()[1]
        self.vocab_size = self.input_y.shape.as_list()[1]
        initializer = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(initializer)
        category_A = int(self.x1.get())
        category_B = int(self.x2.get())
        z_A, z_B = [self.truncated_z_sample(self.num_samples, self.truncation, noise_seed)
                    for noise_seed in [self.noise_seed_A, self.noise_seed_B]]
        y_A, y_B = [self.one_hot([category] * self.num_samples, self.vocab_size)
                    for category in [category_A, category_B]]
        z_interp = self.interpolate_and_shape(z_A, z_B, self.num_interps)
        y_interp = self.interpolate_and_shape(y_A, y_B, self.num_interps)
        ims = self.sample(sess, z_interp, y_interp, self.vocab_size, truncation=self.truncation)
        video_name = 'video.avi'
        height, width, layers = ims[0].shape
        video = cv2.VideoWriter(video_name, 0, 50, (width, height))
        for img in ims:
            video.write(img)
        cv2.destroyAllWindows()
        video.release()
        video = mpe.VideoFileClip(video_name)
        video.write_videofile("output.mp4", fps=50)
        os.startfile("output.mp4")

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        tk.Label(self, text="Img number:").grid(row=0, column=0)
        self.x1 = tk.Entry(self, bd=5)
        self.x1.grid(row=0, column=1)
        tk.Label(self, text="Img number:").grid(row=1, column=0)
        self.x2 = tk.Entry(self, bd=5)
        self.x2.grid(row=1, column=1)
        tk.Button(self, text='Generate!', command=self.create_morphing).grid(row=4, column=0)


import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import tensorflow_addons as tfa
import cv2
from tqdm import tqdm
import numpy as np
from tkinter.messagebox import showerror
import moviepy.editor as mpe
import os

# Load compressed models from tensorflow_hub
class MyModel(tf.keras.Model):
    def __init__(self, mp_sz):
        super(MyModel, self).__init__()
        self.mp_sz = mp_sz
        self.conv1 = tf.keras.layers.Conv2D(64, (5, 5))
        self.act1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.conv2 = tf.keras.layers.Conv2D(64, (5, 5))
        self.act2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.convo = tf.keras.layers.Conv2D((3 + 3 + 2) * 2, (5, 5))

    def call(self, maps):
        x = tf.image.resize(maps, [self.mp_sz, self.mp_sz])
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.convo(x)
        return x


class Morphing(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.TRAIN_EPOCHS = 100
        self.im_sz = 1024
        self.mp_sz = 96
        self.warp_scale = 0.05
        self.mult_scale = 0.4
        self.add_scale = 0.4
        self.add_first = False
        self.fps = 45
        self.steps = 100

        self.top_padding = 50
        self.video = None
        self.image1_file = ''
        self.image2_file = ''
        self.img1_path = os.path.dirname(os.path.abspath(__file__))
        self.img2_path = os.path.dirname(os.path.abspath(__file__))
        tk.Button(self, text='Browse image file', font=44, command=self.browse_1image).place(
            x=60, y=self.top_padding, height=40, width=400)
        self.label0 = tk.Label(self, text='', font=44, background="lightgrey").place(x=460, y=self.top_padding,
                                                                                     height=40, width=400)
        tk.Button(self, text='Browse image file', font=44, command=self.browse_2image).place(
            x=60, y=self.top_padding + 41, height=40, width=400)

        self.label1 = tk.Label(self, text='', font=44, background="lightgrey").place(x=460, y=self.top_padding + 41,
                                                                                 height=40, width=400)
        tk.Button(self, text='Morph', font=44, command=self.create_morphing).place(x=60, y=self.top_padding + 82, height=40, width=400)
        tk.Button(self, text='Save result', font=44, bg='green', command=self.save_morphing).place(
                            x=460, y=self.top_padding + 82, height=40, width=400)
        self.label_slider = tk.Label(self, text='Number of epochs for model training:', font=("TkDefaultFont", 16))
        self.label_slider.place(x=60, y=self.top_padding+123, height=30, width=400)
        self.slider = tk.Scale(self, from_=0, to=1000, orient=tk.HORIZONTAL)
        self.slider.place(x=460, y=self.top_padding+123, height=30, width=400)

    def browse_1image(self):
        self.image1_file = filedialog.askopenfilename(initialdir=self.img1_path,
                                                      title="Select a File",
                                                      filetypes=(("jpeg files", "*.jpg"),
                                                                 ("gif files", "*.gif*"),
                                                                 ("png files", "*.png"),
                                                                 ("all files", "*.*")))
        self.label0 = tk.Label(self, text=os.path.basename(self.image1_file), font=44, background="lightgrey").place(x=460, y=self.top_padding, height=40, width=400)

    def browse_2image(self):
        self.image2_file = filedialog.askopenfilename(initialdir=self.img2_path,
                                                      title="Select a File",
                                                      filetypes=(("jpeg files", "*.jpg"),
                                                                 ("gif files", "*.gif*"),
                                                                 ("png files", "*.png"),
                                                                 ("all files", "*.*")))
        self.label0 = tk.Label(self, text=os.path.basename(self.image2_file), font=44, background="lightgrey").place(x=460, y=self.top_padding + 42, height=40, width=400)

    def save_morphing(self):
        if self.video is None:
            tk.messagebox.showerror(title="Error", message="There is nothing to save. Create morphing first.")
            return

        filename = filedialog.asksaveasfile(initialdir="results", mode='wb', defaultextension=".mp4",
                                            filetypes=(("MP4", "*.mp4"), ("all files", "*.*")))
        if not filename:
            return
        self.video.write_videofile(filename.name, fps=100)

    def create_morphing(self):
        self.TRAIN_EPOCHS = self.slider.get()
        dom_a = cv2.imread(self.image1_file, cv2.IMREAD_COLOR)
        dom_a = cv2.cvtColor(dom_a, cv2.COLOR_BGR2RGB)
        dom_a = cv2.resize(dom_a, (self.im_sz, self.im_sz), interpolation=cv2.INTER_AREA)
        dom_a = dom_a / 127.5 - 1

        dom_b = cv2.imread(self.image2_file, cv2.IMREAD_COLOR)
        dom_b = cv2.cvtColor(dom_b, cv2.COLOR_BGR2RGB)
        dom_b = cv2.resize(dom_b, (self.im_sz, self.im_sz), interpolation=cv2.INTER_AREA)
        dom_b = dom_b / 127.5 - 1

        origins = dom_a.reshape(1, self.im_sz, self.im_sz, 3).astype(np.float32)
        targets = dom_b.reshape(1, self.im_sz, self.im_sz, 3).astype(np.float32)
        self.produce_warp_maps(origins, targets)
        self.use_warp_maps(origins, targets, self.fps, self.steps)

    def create_grid(self, scale):
        grid = np.mgrid[0:scale, 0:scale] / (scale - 1) * 2 - 1
        grid = np.swapaxes(grid, 0, 2)
        grid = np.expand_dims(grid, axis=0)
        return grid

    @tf.function
    def warp(self, origins, targets, preds_org, preds_trg):
        if self.add_first:
            res_targets = tfa.image.dense_image_warp(
                (origins + preds_org[:, :, :, 3:6] * 2 * self.add_scale) * tf.maximum(0.1, 1 + preds_org[:, :, :,
                                                                                          0:3] * self.mult_scale),
                preds_org[:, :, :, 6:8] * self.im_sz * self.warp_scale)
            res_origins = tfa.image.dense_image_warp(
                (targets + preds_trg[:, :, :, 3:6] * 2 * self.add_scale) * tf.maximum(0.1, 1 + preds_trg[:, :, :,
                                                                                          0:3] * self.mult_scale),
                preds_trg[:, :, :, 6:8] * self.im_sz * self.warp_scale)
        else:
            res_targets = tfa.image.dense_image_warp(
                origins * tf.maximum(0.1, 1 + preds_org[:, :, :, 0:3] * self.mult_scale) + preds_org[:, :, :,
                                                                                      3:6] * 2 * self.add_scale,
                preds_org[:, :, :, 6:8] * self.im_sz * self.warp_scale)
            res_origins = tfa.image.dense_image_warp(
                targets * tf.maximum(0.1, 1 + preds_trg[:, :, :, 0:3] * self.mult_scale) + preds_trg[:, :, :,
                                                                                      3:6] * 2 * self.add_scale,
                preds_trg[:, :, :, 6:8] * self.im_sz * self.warp_scale)

        return res_targets, res_origins

    def produce_warp_maps(self, origins, targets):

        model = MyModel(self.mp_sz)

        loss_object = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

        train_loss = tf.keras.metrics.Mean(name='train_loss')

        @tf.function
        def train_step(maps, origins, targets):
            with tf.GradientTape() as tape:
                preds = model(maps)
                preds = tf.image.resize(preds, [self.im_sz, self.im_sz])

                # a = tf.random.uniform([maps.shape[0]])
                # res_targets, res_origins = warp(origins, targets, preds[...,:8] * a, preds[...,8:] * (1 - a))
                res_targets_, res_origins_ = self.warp(origins, targets, preds[..., :8], preds[..., 8:])

                res_map = tfa.image.dense_image_warp(maps, preds[:, :, :,
                                                           6:8] * self.im_sz * self.warp_scale)  # warp maps consistency checker
                res_map = tfa.image.dense_image_warp(res_map, preds[:, :, :, 14:16] * self.im_sz * self.warp_scale)

                loss = loss_object(maps, res_map) * 1 + loss_object(res_targets_, targets) * 0.3 + loss_object(
                    res_origins_, origins) * 0.3

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)

        maps = self.create_grid(self.im_sz)
        maps = np.concatenate((maps, origins * 0.1, targets * 0.1), axis=-1).astype(np.float32)

        epoch = 0
        template = 'Epoch {}, Loss: {}'

        t = tqdm(range(self.TRAIN_EPOCHS), desc=template.format(epoch, train_loss.result()))

        for i in t:
            epoch = i + 1

            t.set_description(template.format(epoch, train_loss.result()))
            t.refresh()

            train_step(maps, origins, targets)

            if (epoch < 100 and epoch % 10 == 0) or \
                    (epoch < 1000 and epoch % 100 == 0) or \
                    (epoch % 1000 == 0):
                preds = model(maps, training=False)[:1]
                preds = tf.image.resize(preds, [self.im_sz, self.im_sz])

                res_targets, res_origins = self.warp(origins, targets, preds[..., :8], preds[..., 8:])

                res_targets = tf.clip_by_value(res_targets, -1, 1)[0]
                res_img = ((res_targets.numpy() + 1) * 127.5).astype(np.uint8)
                cv2.imwrite("images/morph/a_to_b_%d.jpg" % epoch, cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))

                res_origins = tf.clip_by_value(res_origins, -1, 1)[0]
                res_img = ((res_origins.numpy() + 1) * 127.5).astype(np.uint8)
                cv2.imwrite("images/morph/b_to_a_%d.jpg" % epoch, cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))

                np.save('preds.npy', preds.numpy())

    def use_warp_maps(self, origins, targets, fps, steps):
        STEPS = steps

        preds = np.load('preds.npy')

        # save maps as images
        res_img = np.zeros((self.im_sz * 2, self.im_sz * 3, 3))

        res_img[self.im_sz * 0:self.im_sz * 1, self.im_sz * 0:self.im_sz * 1] = preds[0, :, :, 0:3]  # a_to_b add map
        res_img[self.im_sz * 0:self.im_sz * 1, self.im_sz * 1:self.im_sz * 2] = preds[0, :, :, 3:6]  # a_to_b mult map
        res_img[self.im_sz * 0:self.im_sz * 1, self.im_sz * 2:self.im_sz * 3, :2] = preds[0, :, :, 6:8]  # a_to_b warp map

        res_img[self.im_sz * 1:self.im_sz * 2, self.im_sz * 0:self.im_sz * 1] = preds[0, :, :, 8:11]  # b_to_a add map
        res_img[self.im_sz * 1:self.im_sz * 2, self.im_sz * 1:self.im_sz * 2] = preds[0, :, :, 11:14]  # b_to_a mult map
        res_img[self.im_sz * 1:self.im_sz * 2, self.im_sz * 2:self.im_sz * 3, :2] = preds[0, :, :, 14:16]  # b_to_a warp map

        res_img = np.clip(res_img, -1, 1)
        res_img = ((res_img + 1) * 127.5).astype(np.uint8)
        cv2.imwrite("images/morph/maps.jpg", cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))

        # apply maps and save results

        org_strength = tf.reshape(tf.range(STEPS, dtype=tf.float32), [STEPS, 1, 1, 1]) / (STEPS - 1)
        trg_strength = tf.reverse(org_strength, axis=[0])

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video = cv2.VideoWriter('morph.avi', fourcc, fps, (self.im_sz, self.im_sz))
        img_a = np.zeros((self.im_sz, self.im_sz * (STEPS // 10), 3), dtype=np.uint8)
        img_b = np.zeros((self.im_sz, self.im_sz * (STEPS // 10), 3), dtype=np.uint8)
        img_a_b = np.zeros((self.im_sz, self.im_sz * (STEPS // 10), 3), dtype=np.uint8)

        res_img = np.zeros((self.im_sz * 3, self.im_sz * (STEPS // 10), 3), dtype=np.uint8)

        for i in tqdm(range(STEPS)):
            preds_org = preds * org_strength[i]
            preds_trg = preds * trg_strength[i]

            res_targets, res_origins = self.warp(origins, targets, preds_org[..., :8], preds_trg[..., 8:])
            res_targets = tf.clip_by_value(res_targets, -1, 1)
            res_origins = tf.clip_by_value(res_origins, -1, 1)

            results = res_targets * trg_strength[i] + res_origins * org_strength[i]
            res_numpy = results.numpy()

            img = ((res_numpy[0] + 1) * 127.5).astype(np.uint8)
            self.video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            if (i + 1) % 10 == 0:
                res_img[self.im_sz * 0:self.im_sz * 1, i // 10 * self.im_sz: (i // 10 + 1) * self.im_sz] = img
                res_img[self.im_sz * 1:self.im_sz * 2, i // 10 * self.im_sz: (i // 10 + 1) * self.im_sz] = (
                            (res_targets.numpy()[0] + 1) * 127.5).astype(np.uint8)
                res_img[self.im_sz * 2:self.im_sz * 3, i // 10 * self.im_sz: (i // 10 + 1) * self.im_sz] = (
                            (res_origins.numpy()[0] + 1) * 127.5).astype(np.uint8)

        cv2.imwrite("images/morph/result.jpg", cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))

        cv2.destroyAllWindows()
        self.video.release()
        self.video = mpe.VideoFileClip('morph.avi')
        os.startfile("morph.avi")
        print('Result video saved.')

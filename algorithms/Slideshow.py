import tkinter as tk
from tkinter import filedialog
import cv2
import os
import moviepy.editor as mpe
import numpy as np
from algorithms.audioFeatureExtractor import AudioFtExt
from tkinter.messagebox import showerror
import random


class Slideshow(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, highlightbackground="blue", highlightthickness=1)
        self.audio_file = ''
        self.image_folder = ''
        self.final = None
        self.img_folder_path = os.path.dirname(os.path.abspath(__file__))
        self.audio_path = os.path.dirname(os.path.abspath(__file__))
        self.top_padding = 50

        self.folder_button = tk.Button(self, text='Browse folder with images', font=44,
                                       command=self.browse_folder).place(
            x=60, y=self.top_padding + 20, height=40, width=400)
        self.folder_label = tk.Label(self, text='', font=44, background="lightgrey").place(
            x=460, y=self.top_padding + 20, height=40, width=400)

        self.audio_label = tk.Label(self, text='', font=44, background="lightgrey").place(
            x=460, y=self.top_padding + 61, height=40, width=400)
        self.audio_button = tk.Button(self, text='Browse audio file', font=44, command=self.browse_audio).place(
            x=60, y=self.top_padding + 61, height=40, width=400)

        self.submit_button = tk.Button(self, text='Create slideshow', font=44, command=self.create_slideshow).place(
            x=60, y=self.top_padding + 102, height=40, width=400)

        self.save_button = tk.Button(self, text='Save result', font=44, bg='green', command=self.save_slideshow).place(
            x=460, y=self.top_padding + 102, height=40, width=400)

        self.checkboxValue = tk.IntVar()
        self.checkbox = tk.Checkbutton(self, text="Shuffle images", font=44, variable=self.checkboxValue)
        self.checkbox.place(
            x=60, y=self.top_padding + 142, height=40, width=400)

    def save_slideshow(self):
        if self.final is None:
            tk.messagebox.showerror(title="Error", message="There is nothing to save. Create slideshow first.")
            return

        filename = filedialog.asksaveasfile(initialdir="results", mode='wb', defaultextension=".mp4",
                                            filetypes=(("MP4", "*.mp4"),
                                                       ("all files", "*.*")))
        if not filename:
            return
        self.final.write_videofile(filename.name, fps=100)

    # Browse audio file
    def browse_audio(self):
        self.audio_file = filedialog.askopenfilename(initialdir=self.audio_path,
                                                     title="Select a File",
                                                     filetypes=(("wav files",
                                                                 "*.wav"),
                                                                ("all files",
                                                                 "*.*")))
        self.audio_label = tk.Label(self, text=os.path.basename(self.audio_file), font=44,
                                    background="lightgrey").place(
            x=460, y=self.top_padding + 61, height=40, width=400)

    # Browse folder containing photos to generate video from
    def browse_folder(self):
        self.image_folder = filedialog.askdirectory(initialdir=self.img_folder_path, title='Please select a directory')
        self.folder_label = tk.Label(self, text=os.path.basename(self.image_folder), font=44,
                                     background="lightgrey").place(
            x=460, y=self.top_padding + 20, height=40, width=400)

    # Creation of slideshow from folder with photos and audio file
    def create_slideshow(self):
        if self.audio_file == '':
            tk.messagebox.showerror(title="Error", message="Select audio file first.")
            return
        elif self.image_folder == '':
            tk.messagebox.showerror(title="Error", message="Select images folder first.")
            return
        video_name = 'video.avi'
        afe = AudioFtExt(self.audio_file, hz_scale=22050)
        afe.getSpectrogramData()
        afe.getRhythmData(22050, 60)
        beat_times = afe.beat_data
        # Loading images from folder
        images = [img for img in os.listdir(self.image_folder) if
                  img.endswith(".jpeg") or img.endswith(".jpg") or img.endswith(".JPEG") or img.endswith(".JPG")]
        if len(images) == 0:
            tk.messagebox.showerror(title="Error",
                                    message="There are no images in folder: " + self.image_folder)
            return
        frame = cv2.imread(os.path.join(self.image_folder, images[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(video_name, 0, 100, (width, height))
        beat_times = np.append(beat_times, afe.duration_time)
        while(beat_times.size >= len(images)):
            images = images + images
        number_of_frames = int(afe.duration_time * 100)
        image_number = 0
        if self.checkboxValue.get() == 1:
            random.shuffle(images)
        # creation of the video, applying images to the frames
        for i in range(0, number_of_frames):
            # if actual index (actual time) is grater that value corresponding to specific photo, we need to increment image index
            if i / 100 >= beat_times[image_number]:
                image_number = image_number + 1
            if image_number >= len(images):
                break
            img1 = cv2.imread(os.path.join(self.image_folder, images[image_number]))
            img1 = cv2.resize(img1, (width, height), interpolation=cv2.INTER_AREA)
            img2 = cv2.imread(os.path.join(self.image_folder, images[image_number + 1]))
            img2 = cv2.resize(img2, (width, height), interpolation=cv2.INTER_AREA)
            if image_number == 0:
                lastBeat = 0
            else:
                lastBeat = beat_times[image_number - 1]
            # Smooth transition
            diff = (beat_times[image_number] - lastBeat) * 100
            alpha = 1 - (i - lastBeat * 100) / diff
            beta = 1 - alpha
            output = cv2.addWeighted(img1, alpha, img2, beta, 0)
            video.write(output)
        # Mixing music with the picture and creating final video
        cv2.destroyAllWindows()
        video.release()
        audio = mpe.AudioFileClip(self.audio_file)
        video = mpe.VideoFileClip(video_name)
        self.final = video.set_audio(audio)
        self.final.write_videofile("output.mp4", fps=100)
        os.startfile("output.mp4")

import tkinter as tk
from tkinter import filedialog
import cv2
import os
import moviepy.editor as mpe
import numpy as np
from algorithms.audioFeatureExtractor import AudioFtExt


class Slideshow(tk.Frame):
    def browse_audio(self):
        self.audio_file = filedialog.askopenfilename(initialdir="/",
                                                title="Select a File",
                                                filetypes=(("wav files",
                                                            "*.wav"),
                                                           ("all files",
                                                            "*.*")))
        tk.Label(self, text=self.audio_file).grid(row=1, column=0)

    def browse_folder(self):
        self.image_folder = filedialog.askdirectory(initialdir="/", title='Please select a directory')
        tk.Label(self, text=self.image_folder).grid(row=3, column=0)

    def create_slideshow(self):
        video_name = 'video.avi'
        afe = AudioFtExt(self.audio_file, hz_scale=22050)
        afe.getSpectrogramData()
        afe.getRhythmData(22050, 60)
        beat_times = afe.beat_data
        images = [img for img in os.listdir(self.image_folder) if img.endswith(".jpeg") or img.endswith(".jpg") or img.endswith(".JPEG") or img.endswith(".JPG")]
        frame = cv2.imread(os.path.join(self.image_folder, images[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(video_name, 0, 100, (width, height))
        beat_times = np.append(beat_times, afe.duration_time)
        number_of_frames = int(afe.duration_time * 100)
        image_number = 0
        for i in range(0, number_of_frames):
            # if actual index (actual time) is grater that value corresponding to specific photo, we need to increment image index
            if i / 100 >= beat_times[image_number]:
                image_number = image_number + 1
            if image_number >= len(images):
                break
            img1 = cv2.imread(os.path.join(self.image_folder, images[image_number]))
            img2 = cv2.imread(os.path.join(self.image_folder, images[image_number + 1]))
            if image_number == 0:
                lastBeat = 0
            else:
                lastBeat = beat_times[image_number - 1]
            diff = (beat_times[image_number] - lastBeat) * 100
            alpha = 1 - (i - lastBeat * 100) / diff
            beta = 1 - alpha
            output = cv2.addWeighted(img1, alpha, img2, beta, 0)
            video.write(output)
        cv2.destroyAllWindows()
        video.release()
        audio = mpe.AudioFileClip(self.audio_file)
        video = mpe.VideoFileClip(video_name)
        final = video.set_audio(audio)
        final.write_videofile("output.mp4", fps=100)
        os.startfile("output.mp4")

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.audio_file = ''
        self.image_folder = ''
        tk.Button(self, text='Browse audio file', command=self.browse_audio).grid(row=0, column=0)
        tk.Label(self, text='').grid(row=1, column=0)
        tk.Button(self, text='Browse folder with images', command=self.browse_folder).grid(row=2, column=0)
        tk.Label(self, text='').grid(row=3, column=0)
        tk.Button(self, text='Create slideshow', command=self.create_slideshow).grid(row=4, column=0)



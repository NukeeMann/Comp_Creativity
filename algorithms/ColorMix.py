import tkinter as tk
from tkinter import filedialog
import cv2
import os
import moviepy.editor as mpe
import numpy as np
from algorithms.audioFeatureExtractor import AudioFtExt


class ColorMix(tk.Frame):
    def browse_audio(self):
        self.audio_file = filedialog.askopenfilename(initialdir="/",
                                                title="Select a File",
                                                filetypes=(("wav files",
                                                            "*.wav"),
                                                           ("all files",
                                                            "*.*")))
        tk.Label(self, text=self.audio_file).grid(row=1, column=0)

    def generate(self):
        video_name = 'video.avi'
        afe = AudioFtExt(self.audio_file, hz_scale=22050)
        afe.getSpectrogramData()
        afe.getRhythmData(22050, 60)
        spec_data = afe.spec_data.T
        freq = []
        arrSize = int(len(spec_data[0]) / 3)
        for i in range(len(spec_data)):
            freq.append([np.mean(spec_data[i][0:arrSize]), np.mean(spec_data[i][arrSize:2 * arrSize]),
                         np.mean(spec_data[i][2 * arrSize:])])
        max_db = max(freq)
        min_db = min(freq)
        delta_db = [(max_db[0] - min_db[0]), (max_db[1] - min_db[1]), (max_db[2] - min_db[2])]
        width, height = 384, 384
        video = cv2.VideoWriter(video_name, 0, 100, (width, height))
        number_of_frames = int(afe.duration_time * 100)
        for i in range(0, number_of_frames):
            # spectrogram array have different length than there is number of frames, that's why we need to scale index
            current_index = int(i * len(freq) / number_of_frames)
            rgb = freq[current_index]
            rgb = [((rgb[0] - min_db[0]) / delta_db[0]) * 255, ((rgb[1] - min_db[1]) / delta_db[1]) * 255,
                   ((rgb[2] - min_db[2]) / delta_db[2]) * 255]
            rgbImage = np.zeros((width, height, 3), np.uint8)
            # rgbImage[::] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            rgbImage[0:height // 3:] = (int(self.x1.get()), int(self.x2.get()), int(rgb[2]))
            rgbImage[height // 3:2 * height // 3:] = (int(self.x1.get()), int(rgb[1]), int(self.x3.get()))
            rgbImage[2 * height // 3::] = (int(rgb[0]), int(self.x2.get()), int(self.x3.get()))
            video.write(rgbImage)
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
        tk.Button(self, text='Browse audio file', command=self.browse_audio).grid(row=0, column=0)
        tk.Label(self, text='').grid(row=1, column=0)
        self.x1 = tk.Entry(self)
        self.x1.grid(row=2, column=0)
        self.x2 = tk.Entry(self)
        self.x2.grid(row=2, column=1)
        self.x3 = tk.Entry(self)
        self.x3.grid(row=2, column=2)
        tk.Button(self, text='Color Mix', command=self.generate).grid(row=3, column=0)


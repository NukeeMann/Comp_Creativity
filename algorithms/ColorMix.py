import tkinter as tk
from tkinter import filedialog
import cv2
import os
import moviepy.editor as mpe
import numpy as np
from algorithms.audioFeatureExtractor import AudioFtExt
from tkinter.messagebox import showerror


class ColorMix(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.audio_file = ''
        top_padding = 50
        tk.Button(self, text='Browse audio file', font=44, command=self.browse_audio).place(
            x=60, y=top_padding + 20, height=40, width=400)
        self.label1 = tk.Label(self, text='eminem.wav', font=44, background="lightgrey").place(x=460, y=top_padding + 20, height=40, width=400)
        # x1, x2, x3 are components of RGB color which is basis of video, each 1/3 part of frame will be generated from
        # this color with mixed one of those components
        self.x1 = tk.Entry(self, font=44, justify='center')
        self.x1.place(x=60, y=top_padding + 61, height=40, width=266)
        self.x2 = tk.Entry(self, font=44, justify='center')
        self.x2.place(x=327, y=top_padding + 61, height=40, width=266)
        self.x3 = tk.Entry(self, font=44, justify='center')
        self.x3.place(x=594, y=top_padding + 61, height=40, width=266)
        tk.Button(self, text='Color Mix', font=44, command=self.generate).place(x=60, y=top_padding + 102, height=40, width=800)

    # Browse audio file
    def browse_audio(self):
        self.audio_file = filedialog.askopenfilename(initialdir="/",
                                                title="Select a File",
                                                filetypes=(("wav files",
                                                            "*.wav"),
                                                           ("all files",
                                                            "*.*")))
        tk.Label(self, text=self.audio_file).grid(row=1, column=0)

    # Generate video from audio file
    def generate(self):
        if self.audio_file == '':
            tk.messagebox.showerror(title="Error", message="Select audio file first.")
            return
        if self.x1.get() == '' or self.x2.get() == '' or self.x3.get() == '':
            tk.messagebox.showerror(title="Error", message="Select RGB color first.")
            return
        video_name = 'video.avi'
        afe = AudioFtExt(self.audio_file, hz_scale=22050)
        afe.getSpectrogramData()
        afe.getRhythmData(22050, 60)
        # creation of spectrogram data and modifying it to fit our requirements
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
        # creation of the video, applying colors to the frames
        for i in range(0, number_of_frames):
            # spectrogram array have different length than there is number of frames, that's why we need to scale index
            current_index = int(i * len(freq) / number_of_frames)
            rgb = freq[current_index]
            rgb = [((rgb[0] - min_db[0]) / delta_db[0]) * 255, ((rgb[1] - min_db[1]) / delta_db[1]) * 255,
                   ((rgb[2] - min_db[2]) / delta_db[2]) * 255]
            rgbImage = np.zeros((width, height, 3), np.uint8)
            rgbImage[0:height // 3:] = (int(self.x1.get()), int(self.x2.get()), int(rgb[2]))
            rgbImage[height // 3:2 * height // 3:] = (int(self.x1.get()), int(rgb[1]), int(self.x3.get()))
            rgbImage[2 * height // 3::] = (int(rgb[0]), int(self.x2.get()), int(self.x3.get()))
            video.write(rgbImage)
        # mixing music with the picture and creating final video
        cv2.destroyAllWindows()
        video.release()
        audio = mpe.AudioFileClip(self.audio_file)
        video = mpe.VideoFileClip(video_name)
        final = video.set_audio(audio)
        final.write_videofile("output.mp4", fps=100)
        os.startfile("output.mp4")

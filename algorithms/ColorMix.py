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
        self.top_padding = 50
        tk.Button(self, text='Browse audio file', font=44, command=self.browse_audio).place(
            x=60, y=self.top_padding + 20, height=40, width=400)
        self.label1 = tk.Label(self, text='', font=44, background="lightgrey").place(x=460, y=self.top_padding + 20, height=40, width=400)

        self.label_select_color = tk.Label(self, text='Choose basic color (RGB): ', font=("TkDefaultFont", 16))
        self.label_select_color.place(x=60, y=self.top_padding+61, height=60, width=300)

        # x1, x2, x3 are components of RGB color which is basis of video, each 1/3 part of frame will be generated from
        # this color with mixed one of those components
        self.x1 = tk.Entry(self, font=44, justify='center')
        self.x1.place(x=360, y=self.top_padding + 61, height=20, width=100)
        self.x2 = tk.Entry(self, font=44, justify='center')
        self.x2.place(x=360, y=self.top_padding + 81, height=20, width=100)
        self.x3 = tk.Entry(self, font=44, justify='center')
        self.x3.place(x=360, y=self.top_padding + 101, height=20, width=100)

        self.label_frequency_ranges = tk.Label(self, text='Frequency ranges (Hz): ', font=("TkDefaultFont", 16))
        self.label_frequency_ranges.place(x=460, y=self.top_padding+61, height=60, width=300)
        self.freq_range_1 = tk.Entry(self, font=44, justify='center')
        self.freq_range_1.place(x=760, y=self.top_padding + 61, height=30, width=100)
        self.freq_range_2 = tk.Entry(self, font=44, justify='center')
        self.freq_range_2.place(x=760, y=self.top_padding + 91, height=30, width=100)

        tk.Button(self, text='Color Mix', font=44, command=self.generate).place(x=60, y=self.top_padding + 122, height=40, width=400)
        tk.Button(self, text='Save', font=44, command=self.save_color_mix).place(x=460, y=self.top_padding + 122, height=40, width=400)

    def save_color_mix(self):
        if self.final is None:
            tk.messagebox.showerror(title="Error", message="There is nothing to save. Create morphing first.")
            return

        filename = filedialog.asksaveasfile(mode='wb', defaultextension=".mp4", filetypes=(("MP4", "*.mp4"),
                                                                                           ("all files", "*.*")))
        if not filename:
            return
        self.final.write_videofile(filename.name, fps=100)

    # Browse audio file
    def browse_audio(self):
        self.audio_file = filedialog.askopenfilename(initialdir="/",
                                                title="Select a File",
                                                filetypes=(("wav files",
                                                            "*.wav"),
                                                           ("all files",
                                                            "*.*")))
        self.label1 = tk.Label(self, text=os.path.basename(self.audio_file), font=44, background="lightgrey").place(x=460, y=self.top_padding + 20, height=40, width=400)

    # Generate video from audio file
    def generate(self):
        hz_scale = 22050
        if self.audio_file == '':
            tk.messagebox.showerror(title="Error", message="Select audio file first.")
            return
        if self.x1.get() == '' or self.x2.get() == '' or self.x3.get() == '':
            tk.messagebox.showerror(title="Error", message="Select RGB color first.")
            return
        if int(self.x1.get()) > 255 or int(self.x2.get()) > 255 or int(self.x3.get()) > 255 or int(self.x1.get()) < 0 or int(self.x2.get()) < 0 or int(self.x3.get()) < 0:
            tk.messagebox.showerror(title="Error", message="RGB parameters must be in range (0,255).")
            return
        video_name = 'video.avi'
        afe = AudioFtExt(self.audio_file, hz_scale=hz_scale)
        afe.getSpectrogramData()
        afe.getRhythmData(22050, 60)
        # creation of spectrogram data and modifying it to fit our requirements
        spec_data = afe.spec_data.T
        freq = []
        freq_range_1 = int(len(spec_data[0])*int(self.freq_range_1.get())/hz_scale)
        freq_range_2 = int(len(spec_data[0])*int(self.freq_range_2.get())/hz_scale)
        for i in range(len(spec_data)):
            freq.append([np.mean(spec_data[i][0:freq_range_1]), np.mean(spec_data[i][freq_range_1:freq_range_2]),
                         np.mean(spec_data[i][freq_range_2])])
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
        self.final = video.set_audio(audio)
        self.final.write_videofile("output.mp4", fps=100)
        os.startfile("output.mp4")

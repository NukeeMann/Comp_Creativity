import matplotlib.pyplot as plt
import librosa.display


class AudioFtExt:
    audio_path = ""
    hz_scale = 0
    start_time = 0
    duration_time = 0
    audio_data = None
    spec_data = None

    def __init__(self, audio_path, hz_scale=40000, start_time=0.0, duration_time=None):
        self.audio_path = audio_path
        self.hz_scale = hz_scale
        self.start_time = start_time
        self.duration_time = duration_time

    def convertAudioToData(self):
        if self.duration_time is None:
            self.audio_data, self.hz_scale = librosa.load(self.audio_path, sr=self.hz_scale, offset=self.start_time)
            self.duration_time = self.audio_data / self.hz_scale
        else:
            self.audio_data, self.hz_scale = librosa.load(self.audio_path, sr=self.hz_scale, offset=self.start_time,
                                                          duration=self.duration_time)

    def getSpectrogramData(self):
        x = librosa.stft(self.audio_data)
        self.spec_data = librosa.amplitude_to_db(abs(x))

    def plotSpectrogram(self):
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(self.spec_data, sr=self.hz_scale, x_axis='time', y_axis='hz')
        plt.colorbar()
        plt.show()

    def plotWaveplot(self):
        plt.figure(figsize=(14, 5))
        print(type(self.audio_data), type(self.hz_scale))
        librosa.display.waveplot(self.audio_data, sr=self.hz_scale)
        plt.show()

    def samplesPerFrame(self, fps=30):
        # TODO
        # Add function that returns number of samples per frame.
        # spec_data -> [sample number][time stamp][dB Value]
        return
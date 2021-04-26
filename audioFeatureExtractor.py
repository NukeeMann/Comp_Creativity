import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf


class AudioFtExt:
    audio_path = ""  # Path to audio file
    sampling_rate = 0  # Hz upper limit
    start_time = 0  # Number of seconds skipped from the beginning of audio during loading
    duration_time = 0  # Number of seconds loaded from audio
    tempo = 0  # Estimated global tempo in beats per min
    audio_data = None  # Audio data. Loudness of audio in specified time location
    spec_data = None  # Spectrogram data. Loudness of audio in specified frequency frame. (dB)
    beat_data = None  # Beats data. Number of beat and their time location
    cpr_sampling_rate = 0   #
    cpr_audio_data = None

    def __init__(self, audio_path, hz_scale=40000, start_time=0.0, duration_time=None):
        self.audio_path = audio_path
        self.sampling_rate = hz_scale
        self.cpr_audio_path = audio_path
        self.cpr_sampling_rate = hz_scale
        self.start_time = start_time
        self.duration_time = duration_time
        self.convertAudioToData()

    # Loads audio file
    def convertAudioToData(self):
        if self.duration_time is None:
            self.audio_data, self.sampling_rate = librosa.load(self.audio_path, sr=self.sampling_rate,
                                                               offset=self.start_time)
            self.duration_time = len(self.audio_data) / self.sampling_rate
        else:
            self.audio_data, self.sampling_rate = librosa.load(self.audio_path, sr=self.sampling_rate,
                                                               offset=self.start_time, duration=self.duration_time)

    # Creates spectrogram data
    def getSpectrogramData(self):
        x = librosa.stft(self.audio_data)
        self.spec_data = librosa.amplitude_to_db(abs(x))

    # Extracts beat moments
    # ARG 1: Upper limit of Hz spectrum to extract beat drops from
    # ARG 2: Guessed tempo. For better prediction
    def getRhythmData(self, sampling_rate=22050, beats_per_min=60):
        self.cpr_audio_data, self.cpr_sampling_rate = librosa.load(self.audio_path, sr=sampling_rate,
                                                                   offset=self.start_time)

        self.tempo, self.beat_data = librosa.beat.beat_track(self.cpr_audio_data, sr=self.cpr_sampling_rate,
                                                             start_bpm=beats_per_min, units='time')
        return

    # Need to perform getSpectrogramData() and convertAudioToData() first
    def plotSpectrogram(self):
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(self.spec_data, sr=self.sampling_rate, x_axis='time', y_axis='hz')
        plt.colorbar()
        plt.show()

    # Need to perform convertAudioToData() first
    def plotWave(self):
        plt.figure(figsize=(14, 5))
        print(type(self.audio_data), type(self.sampling_rate))
        librosa.display.waveplot(self.audio_data, sr=self.sampling_rate)
        plt.show()

    # Need to perform convertAudioToData() first and getRhythmData() if with_clicks = True
    def saveAudio(self, with_clicks=False, name='test'):
        if with_clicks:
            clicks = librosa.clicks(self.beat_data, sr=self.sampling_rate, length=len(self.audio_data))
            sf.write(name + '.wav', self.audio_data + clicks, self.sampling_rate)
        else:
            sf.write(name + '.wav', self.audio_data, self.sampling_rate)

    def samplesPerFrame(self, fps=30):
        # TODO
        # Add function that returns number of samples per frame.
        # spec_data -> [sample number][time stamp][dB Value]
        return
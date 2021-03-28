from audioFeatureExtractor import AudioFtExt


def main():
    # Define audios path
    audio_path = 'SampleAudio/HzTest.wav'

    # Declare and prepare Audio Feature Extractor class
    afe = AudioFtExt(audio_path)
    # Convert audio to data format
    afe.convertAudioToData()
    # Show plot of loudness of audio
    afe.plotWaveplot()
    # Convert audios data to spectrogram data using Fourier
    afe.getSpectrogramData()
    # Show plot of dB per Hz per second
    afe.plotSpectrogram()
    afe.samplesPerFrame()


if __name__ == "__main__":
    main()

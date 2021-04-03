from audioFeatureExtractor import AudioFtExt


def main():
    # Define audios path
    audio_path = 'SampleAudio/Eminem_the_dfiff.wav'

    # Declare and prepare Audio Feature Extractor class
    afe = AudioFtExt(audio_path)
    # Convert audio to data format
    afe.convertAudioToData()
    # Show plot of loudness of audio
    afe.plotWave()
    # Convert audios data to spectrogram data using Fourier
    afe.getSpectrogramData()
    # Show plot of dB per Hz per second
    afe.plotSpectrogram()
    # Detects beats momentum
    afe.getRhythmData()
    # Saves data to .wav file with CLICK sound at detected beats momentum
    afe.saveAudio(True)


if __name__ == "__main__":
    main()

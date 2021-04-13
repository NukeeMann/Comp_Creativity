from audioFeatureExtractor import AudioFtExt


def main():
    # Define audios path
    audio_path = 'SampleAudio/Eminem_the_dfiff.wav'

    # Declare and prepare Audio Feature Extractor class
    afe = AudioFtExt(audio_path)

    # Convert audio to data format. No need. Already done in __init__ function
    #afe.convertAudioToData()

    # Show plot of loudness of the audio
    afe.plotWave()

    # Convert audios data to spectrogram data using Fourier
    afe.getSpectrogramData()

    # Show plot of dB per Hz per second
    afe.plotSpectrogram()

    # Detects beat
    afe.getRhythmData()

    # Saves data to .wav file with CLICK sound at detected beats momentum
    afe.saveAudio(True)


if __name__ == "__main__":
    main()

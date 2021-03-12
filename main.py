import librosa
import IPython.display as ipd

x, sr = librosa.load("music/eminem2.wav")
ipd.Audio(x, rate=sr)
tempo, beat_times = librosa.beat.beat_track(x, sr=sr, start_bpm=60, units='time')
clicks = librosa.clicks(beat_times, sr=sr, length=len(x))
ipd.Audio(x + clicks, rate=sr)
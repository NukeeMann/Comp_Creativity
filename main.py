import librosa
import cv2
import os
import moviepy.editor as mpe
from audioFeatureExtractor import AudioFtExt
import numpy as np

audio_file = 'music/whitney.wav'
image_folder = 'images'
video_name = 'video.avi'

# default value is 22050. with different value beat times might not be accurate
afe = AudioFtExt(audio_file, hz_scale=22050)
afe.getSpectrogramData()
afe.getRhythmData(60)
beat_times = afe.beat_data

# ## functions that convert spectrogram to one value per one moment in time to easily use it in image procesing;
# TODO: improve
spec_data = afe.spec_data.T
indexes = []
arrSize = int(len(spec_data[0])/3)
for i in range(len(spec_data)):
    indexes.append([np.mean(spec_data[i][0:arrSize]),np.mean(spec_data[i][arrSize:2*arrSize]),np.mean(spec_data[i][2*arrSize:])])
max_db = max(indexes)
min_db = min(indexes)
tmp_db = [(max_db[0] - min_db[0]),(max_db[1] - min_db[1]),(max_db[2] - min_db[2])]
song_time = int(max(beat_times) * 100)
print(song_time)
# create list of images and single frame (all images must be the same size to fit into created frame)
images = [img for img in os.listdir(image_folder) if img.endswith(".JPEG")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 100, (width, height))
# attach images to frames
i = 0.01
frame_number = 0
for image in images:
    while i < beat_times[frame_number]:
        tmp = i * 100 * len(indexes) / song_time
        current_index = int(tmp)  # actual index from flatten spectrogram array
        rgb = indexes[current_index]
        rgb = [((rgb[0]-min_db[0])/tmp_db[0]) * 255,((rgb[1]-min_db[1])/tmp_db[1]) * 255,((rgb[2]-min_db[2])/tmp_db[2]) * 255]
        rgbImage = np.zeros((frame.shape), np.uint8)
        # rgbImage[::] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        rgbImage[::] = (100, 100, int(rgb[2]))
        video.write(rgbImage)
        i = i + 0.01
    frame_number = frame_number + 1
    if frame_number >= len(beat_times):
        break

cv2.destroyAllWindows()
video.release()

audio = mpe.AudioFileClip(audio_file)
video = mpe.VideoFileClip(video_name)
final = video.set_audio(audio)
final.write_videofile("output.mp4", fps=100)
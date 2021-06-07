import librosa
import cv2
import os
import moviepy.editor as mpe
from audioFeatureExtractor import AudioFtExt
import numpy as np

audio_file = 'music/whitney.wav'
image_folder = 'images'
video_name = 'video.avi'

# data initialization
afe = AudioFtExt(audio_file, hz_scale=22050)
afe.getSpectrogramData()
afe.getRhythmData(22050, 60)
beat_times = afe.beat_data

# creation of 2d array containing mean values of specific frequencies for each time stamp
spec_data = afe.spec_data.T
freq = []
arrSize = int(len(spec_data[0])/3)
for i in range(len(spec_data)):
    freq.append([np.mean(spec_data[i][0:arrSize]), np.mean(spec_data[i][arrSize:2*arrSize]), np.mean(spec_data[i][2*arrSize:])])

# values that will be used in scaling
max_db = max(freq)
min_db = min(freq)
delta_db = [(max_db[0] - min_db[0]), (max_db[1] - min_db[1]), (max_db[2] - min_db[2])]

# create list of images and single frame (all images must be the same size to fit into created frame)
images = [img for img in os.listdir(image_folder) if img.endswith(".JPEG")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# creation of video writer
video = cv2.VideoWriter(video_name, 0, 100, (width, height))
# we need to add last 'beat time' so this array covers whole video and not avoid last ~0.5s
beat_times = np.append(beat_times, afe.duration_time)

# attach images to frames
number_of_frames = int(afe.duration_time * 100)
image_number = 0
for i in range(0, number_of_frames):
    # if actual index (actual time) is grater that value corresponding to specific photo, we need to increment image index
    if i/100 >= beat_times[image_number]:
        image_number = image_number + 1
    if image_number >= len(images):
        break
    # spectrogram array have different length than there is number of frames, that's why we need to scale index
    current_index = int(i * len(freq) / number_of_frames)
    rgb = freq[current_index]
    rgb = [((rgb[0] - min_db[0]) / delta_db[0]) * 255, ((rgb[1] - min_db[1]) / delta_db[1]) * 255,
           ((rgb[2] - min_db[2]) / delta_db[2]) * 255]
    # TODO: combine it
    # Change it to False if you want to create video that changes photo to the rhythm
    if False:
        rgbImage = np.zeros(frame.shape, np.uint8)
        # rgbImage[::] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        rgbImage[0:height//3:] = (255, 255, int(rgb[2]))
        rgbImage[height//3:2*height//3:] = (255, int(rgb[1]), 255)
        rgbImage[2*height//3::] = (int(rgb[0]), 255, 255)
        video.write(rgbImage)
    else:
        img1 = cv2.imread(os.path.join(image_folder, images[image_number]))
        img2 = cv2.imread(os.path.join(image_folder, images[image_number + 1]))
        if image_number == 0:
            lastBeat = 0
        else:
            lastBeat = beat_times[image_number-1]
        diff = (beat_times[image_number] - lastBeat)*100
        alpha = 1 - (i - lastBeat*100)/diff
        beta = 1 - alpha
        output = cv2.addWeighted(img1, alpha, img2, beta, 0)
        video.write(output)
        print(alpha)

cv2.destroyAllWindows()
video.release()

audio = mpe.AudioFileClip(audio_file)
video = mpe.VideoFileClip(video_name)
final = video.set_audio(audio)
final.write_videofile("output.mp4", fps=100)

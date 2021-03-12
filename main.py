import librosa
import cv2
import os
import moviepy.editor as mpe

image_folder = 'images'
video_name = 'video.avi'
audio_file = "music/eminem2.wav"

# get rhythm from track
x, sr = librosa.load(audio_file)
tempo, beat_times = librosa.beat.beat_track(x, sr=sr, start_bpm=60, units='time')


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
        video.write(cv2.imread(os.path.join(image_folder, image)))
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
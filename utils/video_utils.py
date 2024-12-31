# Utilities used to read in the video and saving it (CV2)

import cv2

def read_video(video_path): # 24 frame per sec => vid
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret , frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames, output_video_path):
    if not output_video_frames:
        raise ValueError("The list of output video frames is empty.")
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0])) # width x height 
    
    for frame in output_video_frames:
        out.write(frame)
    
    out.release()
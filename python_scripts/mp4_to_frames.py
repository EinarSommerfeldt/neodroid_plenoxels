import cv2
import os

#start_time and end_time in ms
def mp4_to_frames(filepath: str, start_time=-1, end_time=-1):
    folderpath, filename = filepath.rsplit(os.sep, 1)
    foldername = filename.split(".")[0]
    vidcap = cv2.VideoCapture(filepath)
     

    success,image = vidcap.read()
    count = 0
    while success:
        time = vidcap.get(cv2.CAP_PROP_POS_MSEC)
        if time > end_time and end_time != -1:
            break
        if time > start_time:
            os.makedirs(folderpath+os.sep+foldername, exist_ok=True)
            cv2.imwrite(folderpath+os.sep+foldername+os.sep+f"frame{count}.jpg", image)     # save frame as JPEG file    
        success,image = vidcap.read()
        count += 1
    return 1
filepath = r"C:\Users\einar\OneDrive - NTNU\Semester 9\neodroid_plenoxels\svox2\opt\ckpt\fruit_fix\circle_renders.mp4"
mp4_to_frames(filepath, 7000, 11000)
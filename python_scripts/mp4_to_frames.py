import cv2
import os

def mp4_to_frames(filepath: str):
    folderpath, filename = filepath.rsplit(os.sep, 1)
    foldername = filename.split(".")[0]
    vidcap = cv2.VideoCapture(filepath)
    
    success,image = vidcap.read()
    count = 0
    while success:
        os.makedirs(folderpath+os.sep+foldername, exist_ok=True)
        cv2.imwrite(folderpath+os.sep+foldername+os.sep+f"frame{count}.jpg", image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1
    return 1
filepath = r"C:\Users\einarjso\OneDrive - NTNU\Semester 9\Neodroid project\Week 42 doc\test_renders.mp4"
mp4_to_frames(filepath)
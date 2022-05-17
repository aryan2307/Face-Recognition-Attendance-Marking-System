import cv2
import face_recognition as fr
import os
from datetime import datetime
import numpy as np

vid = cv2.VideoCapture(0)
known_encodings = []
names = []
for filename in os.listdir("Training images"):
  img = fr.load_image_file("Training images"+'/'+filename)
  encoding = fr.face_encodings(img)[0]
  known_encodings.append(encoding)
  names.append(filename.split(".")[0])

def makeAttendanceEntry(i):
    with open("Attendance.csv", 'r+', newline = '') as file:
      allLines = file.readlines()
      attendanceList = []
      for line in allLines:
          entry = line.split(',')
          attendanceList.append(entry[0])
      if i not in attendanceList and i != "Image not found":
          now = datetime.now()
          dtString = now.strftime('%d/%b/%Y, %H:%M:%S')
          file.writelines(f'\n{i},{dtString}')


def classify():
  while True: 
    ret, frame = vid.read()
    rgb_frame = frame[:, :, ::-1]
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        result = fr.compare_faces(known_encodings, face_encoding)
        name = "Image not found"
        face_distances = fr.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if result[best_match_index]:
            name = names[best_match_index]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 255, 0), 1)
        makeAttendanceEntry(name)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

import tkinter as tk
from PIL import ImageTk, Image
root= tk.Tk()
canvas1 = tk.Canvas(root, width = 1300, height = 700, background = 'light blue')
canvas1.pack()
label1 = tk.Label(root, text='Face Recognition Attendance Marking System')
label1.config(font = ('Times New Roman', 30))
canvas1.create_window(700, 20, window=label1)

button1 = tk.Button (root, text='Start face recognition',command=classify, bg='orange')
button1.config(height=2, width=20, font = ('Times New Roman', 20))
canvas1.create_window(700, 150, window=button1)

label2 = tk.Label(root, text="Press 'q' key to stop recognition")
label2.config(font = ('Times New Roman', 16))
canvas1. create_window(700, 400, window=label2)

root.mainloop()
vid.release()
cv2.destroyAllWindows()

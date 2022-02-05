import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'image_attendance'
image_list = []
classNames = []

mylist = os.listdir(path)


for cl in mylist:
    currImg = cv2.imread(f'{path}/{cl}')
    image_list.append(currImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(image_list):
    encodeList=[]
    for img_l in image_list:
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img_l)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        print(myDataList)
        nameList=[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dtString}')



encodeListKnown = findEncodings(image_list)
print("encode completes")

cam = cv2.VideoCapture(0)

while True:
    success, img = cam.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25 )
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurrFrame = face_recognition.face_locations(imgS)
    encodesCurrFrame = face_recognition.face_encodings(imgS, faceCurrFrame)

    for ef,fl in zip(encodesCurrFrame, faceCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, ef)
        faceDis = face_recognition.face_distance(encodeListKnown, ef)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = fl
            y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)
import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('elonmusk1.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('elonmusk2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

facLocElon = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (facLocElon[3], facLocElon[0]), (facLocElon[1], facLocElon[2]), (255,0,255), 2 )

facLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (facLocTest[3], facLocTest[0]), (facLocTest[1], facLocTest[2]), (255,0,255), 2 )

res = face_recognition.compare_faces([encodeElon], encodeTest)
print(res)

cv2.imshow('img1', imgElon)
cv2.imshow('img2', imgTest)
cv2.waitKey(0)
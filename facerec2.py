import cv2
import face_recognition
import os
import json
import wget
import requests
import schedule
import shutil
import mediapipe as mp
from datetime import datetime

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
API_URL = 'https://users-api-tifz.onrender.com/users'

vid = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
finger_coord = [(8, 6), (12, 10), (16,14), (20, 18)]
thumb_coord = (4, 2)

known_faces = []
known_names = []
codes = {

}

def load_faces():
    req = requests.get(API_URL)
    data = json.loads(req.text)
    names = []
    for p in data:
        names.append(p['name'])
    for name in os.listdir(KNOWN_FACES_DIR):
        if name not in names:
            shutil.rmtree(f"{KNOWN_FACES_DIR}/{name}")
    for person in data:
        name = person['name']
        url = person['photo_url']
        fing_seq = person['fing_seq']
        codes[name] = fing_seq
        photos = url.split('/*/')
        if name not in os.listdir(KNOWN_FACES_DIR):
            os.mkdir(f"{KNOWN_FACES_DIR}/{name}")
            i = 0
            for photo in photos:
                try:
                    down_image = wget.download(photo, out=f"{KNOWN_FACES_DIR}/{name}")
                    new_name = f"{i}.{down_image.split('.')[1]}"
                    os.rename(down_image, f"{KNOWN_FACES_DIR}/{name}/{new_name}")
                    i += 1
                except:
                    print('an exception occured')
    global known_faces, known_names
    known_faces = []
    known_names = []
    for name in os.listdir(f'{KNOWN_FACES_DIR}'):
        for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
                image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
                encoding = face_recognition.face_encodings(image)[0]
                known_faces.append(encoding)
                known_names.append(name)


schedule.every(1).minutes.do(load_faces)
         
load_faces()
while True:
    schedule.run_pending()
    ret, frame = vid.read()
    
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)


    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            name = known_names[matches.index(True)]

        print(f"{name} is on the screen.\nShow {codes[name]} fingers")
        start = datetime.now()
        start_sec = start.minute * 60 + start.second
        while True:
            ret, frame = vid.read()
    
            small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_small_frame)

            multiLandMarks = results.multi_hand_landmarks
            
            if multiLandMarks:
                handList = []
                for handLms in multiLandMarks:
                    for idx, lm in enumerate(handLms.landmark):
                        cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        handList.append((cx, cy))
                    upCount = 0
                    for coordinate in finger_coord:
                        if handList[coordinate[0]][1] < handList[coordinate[1]][1]:
                            upCount += 1
                    if handList[thumb_coord[0]][0] > handList[thumb_coord[1]][0]:
                        upCount += 1
                print(f"{upCount} fingers up")
            cur = datetime.now()
            cur_sec = cur.minute * 60 + cur.second
            if cur_sec - start_sec > 15:
                break
            

    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.derstroyAllWindows()
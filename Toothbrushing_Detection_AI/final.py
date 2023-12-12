import cv2
import mediapipe as mp
import numpy as np
import time, os
import threading
import librosa
import pygame
from collections import deque
import math

# def rythm():
#     beat_index = 0
#     filename = '산중호걸.wav'
#     y, sr = librosa.load(filename)
#     tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
#     beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
#     pygame.mixer.music.load(filename)
#     pygame.mixer.music.play(1) 

#     start_time = time.time()

#     while pygame.mixer.music.get_busy():
#         current_time = time.time() - start_time
#         #cv2.putText(img, '', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

#         if beat_index < len(beat_times) and current_time >= beat_times[beat_index]:
#             cv2.putText(img, f'Brush!', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
#             beat_index += 1
#             cv2.imshow('img', img)


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

action_seq = []
c_distances = deque(maxlen=60)
checkCircular = ''
checkHeights = []
seq = []
seq_length = 10

mouth_landmark_index = [40, 270, 91, 321]

sizearr = []
inside = False
count = False
toothbrushing = 0
brush = True
even = 0
evenCheck = True
score = 'bad'
score_check = True
timeseq = []

pygame.init()
pygame.mixer.init()

beat_index = 0
filename = '산중호걸.wav'
y, sr = librosa.load(filename)
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

pygame.mixer.music.load(filename)
pygame.mixer.music.play(1) 

# rythm_thread = threading.Thread(target=rythm)
# rythm_thread.daemon = True  # Make the thread a daemon so it exits when the main program exits
# rythm_thread.start()

start_time = time.time()

while cap.isOpened():
    # start_time = time.time()

    ret, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = holistic.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    current_time = time.time() - start_time

    if beat_index < len(beat_times) and current_time >= beat_times[beat_index]:
        cv2.putText(img, f'Brush!', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        cv2.imshow('img', img)
        cv2.putText(img, f'{score}', org=(200, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        if evenCheck == True:
            even += 1
            evenCheck = False
        beat_index += 1
    else:
        if evenCheck == False:
            even += 1
            evenCheck = True

    cv2.putText(img, f'{score}', org=(200, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    
    if result.left_hand_landmarks is not None and result.face_landmarks is not None:

        ''' 희준 part '''
        face_points = result.face_landmarks.landmark
        hand_joints = result.left_hand_landmarks.landmark

        # direction vector (of line equation)
        select_p1 = np.array([(hand_joints[6].x + hand_joints[5].x) / 2, (hand_joints[6].y + hand_joints[5].y) / 2, (hand_joints[6].z + hand_joints[5].z) / 2])
        select_p2 = np.array([(hand_joints[13].x + hand_joints[14].x) / 2, (hand_joints[13].y + hand_joints[14].y) / 2, (hand_joints[13].z + hand_joints[14].z) / 2])
        v = select_p1 - select_p2

        # point (of line equation)
        lower_p1 = np.array([hand_joints[17].x, hand_joints[17].y, hand_joints[17].z])
        lower_p2 = np.array([hand_joints[18].x, hand_joints[18].y, hand_joints[18].z])
        lower_p3 = np.array([hand_joints[19].x, hand_joints[19].y, hand_joints[19].z])
        lower_p4 = np.array([hand_joints[20].x, hand_joints[20].y, hand_joints[20].z])
        p1 = (lower_p1 + lower_p2 + lower_p3 + lower_p4) / 4

        # 칫솔 끝 점
        x1, y1, z1 = hand_joints[6].x, hand_joints[6].y, hand_joints[6].z
        x2, y2, z2 = hand_joints[18].x, hand_joints[18].y, hand_joints[18].z
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        t = (distance*2) / np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        endPoint = [p1[0] + v[0] * t, p1[1] + v[1] * t, p1[2] + v[2] * t]


        # show
        h, w, _ = img.shape
        show_endPoint = (int(endPoint[0] * w), int(endPoint[1] * h))
        show_startPoint = (int(p1[0] * w), int(p1[1] * h))
        cv2.circle(img, show_endPoint, 5, (102, 204, 255), -1)
        cv2.circle(img, show_startPoint, 5, (102, 204, 255), -1)
        cv2.line(img, show_endPoint, show_startPoint, (255, 255, 0), 3)


        # 1. 얼굴가로선-칫솔벡터 사이각
        eyeP1 = np.array([face_points[130].x, face_points[130].y, face_points[130].z])
        eyeP2 = np.array([face_points[359].x, face_points[359].y, face_points[359].z])
        Xaxis = eyeP2 - eyeP1
        angle = np.degrees(np.arccos(np.dot(Xaxis, v) / (np.linalg.norm(Xaxis) * np.linalg.norm(v))))
        
        '''
        # 2. 화면 가로선-칫솔벡터 사이각
        vFixed = [1,0]
        angleFixed = np.degrees(np.arccos(np.dot(vFixed, v[:2]) / (np.linalg.norm(vFixed) * np.linalg.norm(v[:2]))))
        '''         


        # 얼굴세로선-endpoint 좌우 구분
        Yaxis = face_points[168].x  # 미간의 x좌표
        Xinterval = endPoint[0] - Yaxis


        # check if it is left or right or mid
        if 165 < angle:
            action = 'mid horizontal'
        else:
            if 85 < angle < 110 and face_points[39].x < endPoint[0] < face_points[269].x:
                if v[1] > 0:
                    action = 'mid vertical lower'
                else:
                    action = 'mid vertical upper'

            else:
                if Xinterval > 0:
                    action = 'right'
                else:
                    action = 'left'


        # check if it is circular
        c_distance = math.sqrt((p1[0]-face_points[168].x)**2 + (p1[1]-face_points[168].y)**2)   # 미간점
        c_distances.append(c_distance)

        if len(c_distances) >= 60:
            mean_distance = sum(c_distances) / len(c_distances)
            abs_distances = [abs(mean_distance-c_distance) for c_distance in c_distances]
            abs_distances.sort()
            abs_distances = deque(abs_distances)
            for _ in range(10):
                abs_distances.pop()
                abs_distances.popleft()
            abs_distances = list(abs_distances)
            abs_distance = sum(abs_distances)
            print(abs_distance)
            if action == 'mid vertical lower' or action == 'mid vertical upper':
                checkCircular = 'Linear'
            elif action == 'mid horizontal':
                checkCircular = 'Circular'
            else:
                if abs_distance > 0.53:
                    checkCircular = 'Circular'
                else:
                    checkCircular = 'Linear'
        

        # cehck if it is upper or lower
        if (action == 'right' or action == 'left') and checkCircular == 'Linear':
            mid = (face_points[138].y+face_points[367].y) / 2   # 입술중간점 양 옆 볼쪽 landmark
            checkHeights.append(mid-endPoint[1])
            if checkHeights:
                if sum(checkHeights)/len(checkHeights) > 0:
                    action += ' upper'
                else:
                    action += ' lower'

        # output
        action_seq.append(action)

        if len(action_seq) < 19:
            continue

        this_action = '?'
        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            this_action = action

        cv2.putText(img, f'{checkCircular.upper()}', org=(20, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
        cv2.putText(img, f'{this_action.upper()}', org=(20, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    





        ''' 예준 part '''
        vm0 = np.array([face_points[mouth_landmark_index[0]].x, face_points[mouth_landmark_index[0]].y, face_points[mouth_landmark_index[0]].z])
        vm1 = np.array([face_points[mouth_landmark_index[1]].x, face_points[mouth_landmark_index[1]].y, face_points[mouth_landmark_index[1]].z])
        vm2 = np.array([face_points[mouth_landmark_index[2]].x, face_points[mouth_landmark_index[2]].y, face_points[mouth_landmark_index[2]].z])
        vm3 = np.array([face_points[mouth_landmark_index[3]].x, face_points[mouth_landmark_index[3]].y, face_points[mouth_landmark_index[3]].z])

        n0 = np.cross(vm1-vm0, vm2-vm0)
        n1 = np.cross(vm3-vm1, vm0-vm1)
        n2 = np.cross(vm0-vm2, vm3-vm2)
        n3 = np.cross(vm2-vm3, vm1-vm3)

        horizontal_left = vm0-vm1
        horizontal_right = vm1-vm0
        horizontal_left = horizontal_left/np.linalg.norm(horizontal_left)
        horizontal_right = horizontal_right/np.linalg.norm(horizontal_right)

        n0 = n0/np.linalg.norm(n0)
        n1 = n1/np.linalg.norm(n1)
        n2 = n2/np.linalg.norm(n2)
        n3 = n3/np.linalg.norm(n3)

        n = [n0, n1, n2, n3]

        vm0 = vm0 + n0*0.1 + horizontal_left*0.05
        vm1 = vm1 + n1*0.1 + horizontal_right*0.05
        vm2 = vm2 + n2*0.1 + horizontal_left*0.05
        vm3 = vm3 + n3*0.1 + horizontal_right*0.05

        v_array = [vm0, vm1, vm2, vm3, np.array([face_points[mouth_landmark_index[0]].x, face_points[mouth_landmark_index[0]].y, face_points[mouth_landmark_index[0]].z]),
                                       np.array([face_points[mouth_landmark_index[1]].x, face_points[mouth_landmark_index[1]].y, face_points[mouth_landmark_index[1]].z]),
                                       np.array([face_points[mouth_landmark_index[2]].x, face_points[mouth_landmark_index[2]].y, face_points[mouth_landmark_index[2]].z]),
                                       np.array([face_points[mouth_landmark_index[3]].x, face_points[mouth_landmark_index[3]].y, face_points[mouth_landmark_index[3]].z])]
        dist = []

        brushHead = np.array([endPoint[0], endPoint[1], endPoint[2]])

        for i in range(len(v_array)):
            dist.append(v_array[i] - brushHead)

        # display dots for vm0, vm1, vm2, vm3 on the img
        # for i in range(len(v_array)):
        #     if (dist[i][0]*dist[i][0] + dist[i][1]*dist[i][1] + dist[i][2]*dist[i][2]) < 0.005:
        #         cv2.circle(img, (int(v_array[i][0]*img.shape[1]), int(v_array[i][1]*img.shape[0])), 5, (0, 255, 0), -1)
        #     else:
        #         cv2.circle(img, (int(v_array[i][0]*img.shape[1]), int(v_array[i][1]*img.shape[0])), 5, (255, 0, 0), -1)
      

        x_avg = endPoint[0]
        y_avg = endPoint[1]
        z_avg = endPoint[2]

        d = [x_avg, y_avg, z_avg]
        size = d[0]*d[0] + d[1]*d[1] + d[2]*d[2]
        sizearr.append(size)
        seq.append(d)
        sizearr = sizearr[-seq_length:]

        if size < np.min(sizearr) + (np.max(sizearr)-np.min(sizearr))*0.3:
            inside = True
        else:
            inside = False
            count = False
        
        if inside==True:
            if count==False:
                toothbrushing += 1
                count = True

        if even%4 == 1 or even%4 == 2: # brushing time
            if score_check == True:
                if size < np.min(sizearr) + (np.max(sizearr)-np.min(sizearr))*0.3:
                    score = 'perfect!'
                    score_check = False
                elif size < np.min(sizearr) + (np.max(sizearr)-np.min(sizearr))*0.6:
                    score = 'good'
                    score_check = False
                else:
                    score = 'bad'
                    score_check = False
        else:
            score_check = True
                
            
        # # 1. Draw face landmarks
        # mp_drawing.draw_landmarks(img, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
        #                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        #                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        #                             )
        # # 3. Left Hand
        # mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
        #                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        #                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        #                             )
        
        if len(seq) < seq_length:
            continue
        
        # 프레임 처리 종료 시간 기록
        # end_time = time.time()

        # 프레임당 latency 계산
        # latency = end_time - start_time
        # print(f"Latency for this frame: {latency} seconds")
        # timeseq.append(latency)
        # timeseq[-seq_length:]

        cv2.putText(img, f'{toothbrushing}', org=(int(result.left_hand_landmarks.landmark[0].x * img.shape[1]), int(result.left_hand_landmarks.landmark[0].y * img.shape[0] + 40)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break
    
    # print(f"Average latency for {seq_length} frames: {np.mean(timeseq[-seq_length:])} seconds")
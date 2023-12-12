import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import time, os

'''''''''''create_dataset.py 로직과 동일'''''''''''


actions = ['CIRCLE_1', 'CIRCLE_2', 'CIRCLE_3', 'HORIZONTAL_1', 'HORIZONTAL_2', 'HORIZONTAL_3', 'HORIZONTAL_4', 'VERTICAL_1', 'VERTICAL_2'] # idx 0, 1, 2 매칭
seq_length = 30
tf.keras.utils.disable_interactive_logging()
model = load_model('models\model.h5')

# MediaPipe holyster model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out = cv2.VideoWriter('input.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
# out2 = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

seq = []
action_seq = []

# face -> 눈 주위 가로선 landmarks
face_landmark_indices = [127, 34, 143, 35, 226, 130, 33, 7, 163, 144, 145, 153, 154, 155, 133, 243, 244, 245, 122, 6, 351, 465, 464, 463, 362, 382, 381, 380, 374, 373, 390, 249, 263, 369, 446, 265, 372, 264, 356]
hand_landmark_index = [9]

sizearr = []
inside = False
count = False
toothbrushing = 0
brush = True
even = 0
score = 'bad'
score_check = True

while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = holistic.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if brush==True:
        start_time = time.time()
        brush = False
        even += 1
    
    cv2.putText(img, f'{score}', org=(50, 350), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    if even%2 == 1:
        if time.time() - start_time < 0.55:
            cv2.putText(img, f'brush!', org=(50, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        else:
            brush = True    
    elif even%2 == 0:
        if time.time() - start_time < 0.01:
            cv2.putText(img, f'', org=(50, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        else:
            score_check = True
            brush = True

        

    if result.left_hand_landmarks is not None and result.face_landmarks is not None:
        
        face_point = np.zeros((468, 4))
        hand_joint = np.zeros((21, 4))

        for j, lm in enumerate(result.face_landmarks.landmark):
            face_point[j] = [lm.x, lm.y, lm.z, lm.visibility] # see or not - visibility


        for k, lm2 in enumerate(result.left_hand_landmarks.landmark):
            hand_joint[k] = [lm2.x, lm2.y, lm2.z, lm2.visibility] # see or not - visibility

        # Compute angles between joints
        v1 = face_point[face_landmark_indices, :3]
        v2 = hand_joint[hand_landmark_index, :3]
        v = v1 - v2 # [39, 3]
        '''
        # Normalize v
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Get angle using arcos of dot product
        angle = np.arccos(np.einsum('nt,nt->n',
            v[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37],:], 
            v[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38],:]))

        angle = np.degrees(angle) # Convert radian to degree

        d = np.concatenate([face_point.flatten(), hand_joint.flatten(), angle])
        '''
        x_avg = np.float32(0)
        y_avg = np.float32(0)
        z_avg = np.float32(0)

        for i in range(39):
            x_avg += v[i][0]
            y_avg += v[i][1]
            z_avg += v[i][2]
            
        x_avg /= 39
        y_avg /= 39
        z_avg /= 39

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

        if even%2 == 1: # brushing time
            if score_check == True:
                if size < np.min(sizearr) + (np.max(sizearr)-np.min(sizearr))*0.1:
                    score = 'perfect!'
                    score_check = False
                elif size < np.min(sizearr) + (np.max(sizearr)-np.min(sizearr))*0.5:
                    score = 'good'
                    score_check = False
        elif even%2 == 0:
            if score_check == True:
                score = 'bad'

            
        # print(toothbrushing)
        # print(d)
        # print(np.max(sizearr))
        # print(np.min(sizearr))

        '''
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(img, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                    )
        # 3. Left Hand
        mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                    )
        '''
        # if len(seq) < seq_length:
        #     continue

        # input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

        # y_pred = model.predict(input_data).squeeze()

        # i_pred = int(np.argmax(y_pred))
        # conf = y_pred[i_pred]

        # if conf < 0.9:
        #     continue

        # action = actions[i_pred]
        # action_seq.append(action)

        # if len(action_seq) < 9:
        #     continue

        # this_action = '?'
        # if action_seq[-1] == action_seq[-2] == action_seq[-3]:
        #     this_action = action

        # cv2.putText(img, f'{this_action.upper()}', org=(int(result.left_hand_landmarks.landmark[0].x * img.shape[1]), int(result.left_hand_landmarks.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.putText(img, f'{toothbrushing}', org=(int(result.left_hand_landmarks.landmark[0].x * img.shape[1]), int(result.left_hand_landmarks.landmark[0].y * img.shape[0] + 40)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # out.write(img0)
    # out2.write(img)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

    # print(sizearr)
    # print(np.max(sizearr))
    # print(np.min(sizearr))
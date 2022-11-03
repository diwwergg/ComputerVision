import cv2
import time
import numpy as np

cap = cv2.VideoCapture('6352300197.avi')
starting_time = time.time()
frame_id = 1
prev_frame = None
L = 50
L_thick = 10
R = -50
R_thick = 10
iframe = 0

L_count = 0
R_count = 0
In_people = 0
Out_people = 0

L_on = True
R_on = True
while True:
    frame_id += 1
    iframe += 1
    _, frame = cap.read()
    L_act = False
    R_act = False
    if prev_frame is not None and iframe > 1:
        diffL = cv2.absdiff(prev_frame[:, L:L + L_thick], frame[:, L:L + L_thick])
        diffR = cv2.absdiff(prev_frame[:, R:R + R_thick], frame[:, R:R + R_thick])
        # IN
        if diffL.sum() > 800:
            L_act = True
            L_count = L_count + 1
        else:
            if (L_count > 0):
                if L_on is True and R_on is False:
                    R_on = True
                    Out_people = Out_people + 1
                    iframe = 0
                    print('In:', In_people, ' Out:', Out_people)
                elif L_on is True and R_on is True:
                    L_on = False

                print('R:', R_on, '  L:', L_on)
            L_count = 0

        # OUT
        if diffR.sum() > 8000:
            R_act = True
            R_count = R_count + 1
        else:
            if (R_count > 0):
                if L_on is False and R_on is True:
                    L_on = True
                    In_people = In_people + 1
                    iframe = 0
                    print('Out:', Out_people, ' In:', In_people)
                elif L_on is True and R_on is True:
                    R_on = False

                print('R:', R_on, '  L:', L_on)
            R_count = 0

        # cv2.imshow('diffR', diffR)

    prev_frame = frame.copy()
    if L_act:
        frame[:, L:L + L_thick, :2] = 0
    else:
        frame[:, L:L + L_thick, [0, 2]] = 0
    if R_act:
        frame[:, R:R + R_thick, :2] = 0
    else:
        frame[:, R:R + R_thick, [0, 2]] = 0
    frame = cv2.putText(frame, ('In :' + str(In_people)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA, False)
    frame = cv2.putText(frame, ('Out :' + str(Out_people)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA, False)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    if ((L_on is False or R_on is False) and iframe > 300):
        L_on = True
        R_on = True
        print('clear  R:', R_on, '  L:', L_on)
        iframe = 0
print ('in')

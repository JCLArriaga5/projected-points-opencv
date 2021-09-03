import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import *

if __name__ == '__main__':
    # Calibration steo
    patternsize = (9, 6)
    port = 2
    print('''Calibration Step''')
    path = takeimages(opencam(port))
    objpoints, imgpoints = findpattern('chessboard', patternsize, path, True)
    mtx, dist = getintrinsic(objpoints, imgpoints)

    # Prepare variables
    cam = opencam(port)
    if not cam.isOpened():
        print("Cannot open camera")
        exit()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
    r, c = patternsize
    objp = np.zeros((r * c, 3), np.float32)
    objp[:, :2] = np.mgrid[0:r, 0:c].T.reshape(-1, 2)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        key = cv2.waitKey(1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected, corners = cv2.findChessboardCorners(gray, patternsize, None)

        if detected:
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            _, rvec, tvec, _ = cv2.solvePnPRansac(objp, corners, mtx, dist)
            R, _ = cv2.Rodrigues(rvec)
            P = mtx.dot(np.hstack([R, tvec]))

            x = y = z = 0

            draw_axes(frame, R, tvec, mtx, dist)
            for i in range(0, patternsize[0] - 1, 2):
                for j in range(0, patternsize[1] - 1, 2):
                        draw_cube(frame, P, (x + i), (y + j), z, 1, 1, (163, 16, 117))

        cv2.imshow('Realtime', frame)

        if key%256 == 27:
            # Esc pressed
            break

    cam.release()
    cv2.destroyAllWindows()

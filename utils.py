# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
import glob
import os

def opencam(port):
    '''
    Open camera using DirectShow (via videoinput).

    Parameters
    ----------
    port : Device index, is just the number to specify which camera.
    '''

    return cv2.VideoCapture(port, cv2.CAP_DSHOW)

def numcams():
    '''
    Show the number of available devices, their width and height and the port number.

    Return
    ------
    Number of devices availables.
    '''

    port = 0
    device_inf = {}
    while True:
        cam = opencam(port)
        if not cam.isOpened():
            break

        ret, frame = cam.read()
        if ret:
            device_inf['Device {}'.format(port + 1)] = [port, cam.get(3), cam.get(4)]

        port += 1

    for k, v in sorted(device_inf.items()):
        print('{} : port = {}, w = {}, h = {}'.format(k, v[0], v[1], v[2]))

    return int(len(device_inf))

def testdevice(port):
    '''
    Test the device, to see if it works correctly.

    Parameters
    ----------
    port : Device index, is just the number to specify which camera.
    '''

    cam = opencam(port)
    if not cam.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imshow('test cam in port {}'.format(port), frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

def takephoto(cam, cnt, path):
    '''
    Take a photo of a video frame.

    Parameters
    ----------
    cam : cv2.VideoCapture object, you can use the opencam function.
    cnt : int
        Number you want the frame capture to have.
    path : str
        Desired directory where you want it to be saved.
    '''

    ret, frame = cam.read()
    if ret:
        cv2.imwrite(path + '/img' + '%06d.png'%(cnt), frame)

def takeimages(cam):
    '''
    Image capture for the camera calibration stage.
    The images are stored in a directory named after the date they were taken.

    Parameters
    ----------
    cam : cv2.VideoCapture object, you can use the opencam function.
    '''

    path = './images/calibration/' + datetime.now().strftime('%d-%m-%Y')
    if not os.path.exists(path):
        os.makedirs(path)

    print('Press Esc to finish.')
    print('Press Space to take a picture.')

    cnt = 1
    if not cam.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        key = cv2.waitKey(1)
        cv2.imshow('Take calibration images', frame)

        if key%256 == 27:
            # Esc pressed
            break
        elif key%256 == 32:
            # Space pressed
            takephoto(cam, cnt, path)
            cnt += 1

    cam.release()
    cv2.destroyAllWindows()

    return path

def img640x480(img):
    '''
    Resize image to 640x480.

    Parameters
    ----------
    img : Image to resize.
    '''

    return cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

def imgpath_inf(imgpath):
    '''
    Create dictionary that contains information about the images within a directory.

    Parameters
    ----------
    imgpath : str
        Path of the directory from which you want to know the information of the images.

    Return
    ------
    Directory with images information (Number, Name, widh, height, type).
    '''

    img_inf = {}
    if glob.glob(imgpath + '/*.png') or glob.glob(imgpath + '/*.jpg'):
        fnames = os.listdir(imgpath)
        inf = [cv2.imread(imgpath + '/' + n).shape for n in fnames]

        for n in range(len(inf)):
            if len(inf[n]) > 2:
                type = 'color'
            else:
                type = 'grayscale'

            img_inf['{}: {}'.format(n + 1, fnames[n])] = [inf[n][1], inf[n][0], type]

    return img_inf

def show_imgpath_inf(dic):
    '''
    Print information about the images.
    '''

    if len(dic) == 0:
        raise ValueError('Dictionary empty')

    print('{:^24s}'.format('Image information for camera calibration'))
    for k, v in dic.items():
        print('|    {} : w = {}, h = {}, type = {}'.format(k, v[0], v[1], v[2]))

def findpattern(pattern, patternsize, imgpath, display=True):
    '''
    Finding the corners in the chess pattern for the camera calibration stage.

    Parameters
    ----------
    pattern : str
        Name of pattern e.g. chessboard.
    patternsize : tuple
        Tuple that contains the size to detect in pattern.
    imgpath : str
        Images path for calibration.
    display : bool
        True if you want to see the corners detections in each image.

    Returns
    -------
    objpoints : 3d point in real world space.
    imgpoints : 2d points in image plane.
    '''

    if type(patternsize) is not tuple:
        raise ValueError('size must be in tuple format')
    if pattern not in ['chessboard']:
        raise ValueError('Valid patterns: chessboard')

    if glob.glob(imgpath + '/*.png'):
        imgs = glob.glob(imgpath + '/*.png')
    elif glob.glob(imgpath + '/*.jpg'):
        imgs = glob.glob(imgpath + '/*.jpg')
    else:
        raise ValueError('image format not supporting')

    show_imgpath_inf(imgpath_inf(imgpath))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
    r, c = patternsize
    objp = np.zeros((r * c, 3), np.float32)
    objp[:, :2] = np.mgrid[0:r, 0:c].T.reshape(-1, 2)
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for fname in imgs:
        img = cv2.imread(fname)

        if img.shape[:2] != (480, 640):
            print('Resize image to 640x480')
            img = img640x480(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, patternsize, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            if display:
                cv2.drawChessboardCorners(img, patternsize, corners2, ret)
                cv2.imshow('Corners detected', img)
                print('Press space...')
                cv2.waitKey(-1)
        else:
            print("[ERROR] Could not perform the detection")

    cv2.destroyAllWindows()

    return objpoints, imgpoints

def getintrinsic(objpoints, imgpoints, img_size=(640, 480)):
    '''
    Obtain intrinsic and distortion parameters of the camera.

    Parameters
    ----------
    objpoints : array
        3d point in real world space.
    imgpoints : array
        2d points in image plane.
    img_size : tuple
        Size of the image.

    Returns
    -------
    mxt : Intrinsic parameters.
    dist : Distortion parameters.
    '''

    _, mxt, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mxt, dist

def circle_pts(r, cx, cy, num_pts):
    # Homogeneous form
    return [[r * np.cos(theta) + cx, r * np.sin(theta) + cy, 0, 1] for theta in np.linspace(0, 2 * np.pi, num_pts)]

def projpts(pts, P):
    projected = P.dot(pts)
    w = projected[2, :]
    projected /= w

    return projected

def draw_axes(frame, R, t, mtx, dist):
    xyzo = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1], [0, 0, 0]]).reshape(-1,3)

    axis, _ = cv2.projectPoints(xyzo, R, t, mtx, dist)
    axis = axis.astype(int)
    cv2.line(frame, tuple(axis[3].ravel()),
                    tuple(axis[0].ravel()), (255, 0, 0), 3)
    cv2.line(frame, tuple(axis[3].ravel()),
                    tuple(axis[1].ravel()), (0, 255, 0), 3)
    cv2.line(frame, tuple(axis[3].ravel()),
                    tuple(axis[2].ravel()), (0, 0, 255), 3)

def draw_cube(frame, P, x, y, z, w, h, color):
    pts = np.float32([
        [x, y, -z, 1], [x + w, y, -z, 1], [x + w, y + w, -z, 1], [x, y + w, -z, 1],
        [x, y, -(z + h), 1], [x + w, y, -(z + h), 1],
        [x + w, y + w, -(z + h), 1], [x, y + w, -(z + h), 1]
    ]).reshape(-1, 4).transpose()

    projected = projpts(pts, P)
    under = projected[:, :4]
    over = projected[:, 4:]

    under_fill = np.array(
        [[int(under[0, i]), int(under[1, i])] for i in range(under.shape[1])]
    )

    cv2.fillPoly(frame, pts = [under_fill], color=color)

    [cv2.line(frame, (int(under[0, i]), int(under[1, i])),
                     (int(under[0, i - 1]), int(under[1, i - 1])), color, 2)
        for i in range(under.shape[1])]

    [cv2.line(frame, (int(over[0, i]), int(over[1, i])),
                     (int(over[0, i - 1]), int(over[1, i - 1])), color, 2)
        for i in range(over.shape[1])]

    [cv2.line(frame, (int(under[0, i]), int(under[1, i])),
                     (int(over[0, i]), int(over[1, i])), color, 2)
        for i in range(under.shape[1])]

if __name__ == '__main__':
    path = takeimages(opencam(0))

    objpoints, imgpoints = findpattern('chessboard', (9, 6), path, True)
    mxt, dist = getintrinsic(objpoints, imgpoints)
    print('Intrinsic matrix')
    print(mxt)

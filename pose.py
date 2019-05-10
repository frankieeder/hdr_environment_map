import cv2
import numpy as np
import os

def pose(imgs):
    objectPoints = [get_3d_checkerboard_points() for _ in imgs]
    imagePoints = [get_2d_checkerboard_points(img) for img in imgs]
    test_img = np.array(imgs[0])
    for pt in imagePoints[0]:
        y, x = pt.astype(int)
        test_img[x:20+x, y:20+y, :] = (255, 0, 0)
    cv2.imwrite("points.png", test_img)
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=objectPoints,
        imagePoints=imagePoints,
        imageSize=imgs[0].shape[:2],
        cameraMatrix=None,
        distCoeffs=None
    )
    return retval, cameraMatrix, distCoeffs, rvecs, tvecs

def get_3d_checkerboard_points():
    x = np.linspace(1, 9, 9)  # Starts one block in from the left and top, so start at 1
    y = np.linspace(1, 9, 9)
    z = np.zeros((9, 9))
    xy = np.meshgrid(x, y)
    xyz = np.array(xy + [z])
    xyz_coords = np.moveaxis(xyz, 0, 2)
    xyz_coord_list = xyz_coords.reshape(-1, 3)
    return xyz_coord_list.astype(np.float32)

def get_2d_checkerboard_points(img, pattern_size=(9,9)):
    umat = cv2.UMat(img)
    retval, corners = cv2.findChessboardCorners(umat, pattern_size)
    return corners.get()[:, 0, :]

calibration_imgs_dir = "images/calibration"
files = [os.path.join(calibration_imgs_dir, f) for f in os.listdir(calibration_imgs_dir)]
calibration_imgs = [cv2.imread(f) for f in files]
pose = pose(calibration_imgs)
pass

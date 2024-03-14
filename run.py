from util import resize_image, HWC3
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

model_openpose = None


def openpose(img, res, hand_and_face):
    img = resize_image(HWC3(img), res)
    global model_openpose
    if model_openpose is None:
        from openpose import OpenposeDetector
        model_openpose = OpenposeDetector()
    result = model_openpose(img, hand_and_face)
    return [result]

def openpose_pose(img):
    global model_openpose
    if model_openpose is None:
        from openpose import OpenposeDetectorBodyOnly
        model_openpose = OpenposeDetectorBodyOnly()
    pose_pred, pose_img = model_openpose(img, return_is_index=True)
    return pose_pred, pose_img
        
input_path = 'PATH to your image file'

test_img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)

# get predicted openpose; aligned pose images
pose_img, pred_pose = openpose_pose(test_img)

# you can visualize ...
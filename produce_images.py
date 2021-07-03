from face_detector import FaceDetector
from PIL import Image
from utils import landmarks_distance, correct_detection, angle_between_2_points, extract_left_eye_center, extract_right_eye_center, crop_image, draw_axis
from torch.autograd import Variable
from torchvision import transforms
import cv2
import dlib
import hopenet
import json
import numpy as np
import os
import pickle
import random
from skimage import transform as tf
import sys
import torch
import torch.nn.functional as F
import torchvision

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib/shape_predictor_68_face_landmarks.dat")
TEMPLE_L = 0
TEMPLE_R = 16
EYEBROW_LIMIT_L = 17
EYEBROW_LIMIT_R= 26
NOSE_TIP = 33
EYE_LIMIT_L = 36
EYE_LIMIT_R = 45
NOSE_CENTER = 27
EYEBROW_TOP_L = 19
EYEBROW_TOP_R = 24
EYE_BOTTOM_L = 40
EYE_BOTTOM_R = 46

class RPYEstimator:

    def __init__(self, snapshot_path, gpu=0):
        self.model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        saved_state_dict = torch.load(snapshot_path)
        self.model.load_state_dict(saved_state_dict)
        self.transformations = transforms.Compose([transforms.Scale(224),
                                                transforms.CenterCrop(224), transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.gpu = gpu
        self.model.cuda(gpu)
        self.model.eval()
        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).cuda(gpu)

    def correct_detection_hopenet(self, face, img):
        left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
        bbox_width = abs(right - left)
        bbox_height = abs(bottom - top)

        left -= 2 * bbox_width // 4
        if left < 0:
            left = 0
        
        top -= 3 * bbox_height // 4
        if top < 0: 
            top = 0

        right += 2 * bbox_width // 4
        if right > img.shape[1]:
            right = img.shape[1]

        bottom += bbox_height // 4
        if bottom > img.shape[0]:
            bottom = img.shape[0]
        return img[top:bottom, left:right]

    def estimate(self, face, img):
        img = self.correct_detection_hopenet(face, img)
        img = Image.fromarray(img)
        img = self.transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = Variable(img).cuda(self.gpu)

        yaw, pitch, roll = self.model(img)

        yaw_predicted = F.softmax(yaw)
        pitch_predicted = F.softmax(pitch)
        roll_predicted = F.softmax(roll)

        yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99
        return roll_predicted, pitch_predicted, yaw_predicted

class Styler:
    
    def __init__(self, glasses_path, hats_path, scarves_path, scarves_points="scarf_points.json"):
        self.glasses = []
        self.hats = []
        for img in os.listdir(glasses_path):
            self.glasses.append(Image.open(os.path.join(glasses_path, img)))
        for img in os.listdir(hats_path):
            self.hats.append(Image.open(os.path.join(hats_path, img)).convert("RGBA"))
        
        points = json.load(open(os.path.join(scarves_path, scarves_points), "r"))
        self.scarves = dict()
        for k, v in points.items():
            self.scarves[k] = {
                "image": Image.open(os.path.join(scarves_path, "{}.png".format(k))).convert("RGBA"),
                "points": v
            }

    def put_scarf(self, face_img, landmarks):
        scarf_index = random.choice(list(self.scarves.keys()))
        scarf = self.scarves[scarf_index]
        face_img = Image.fromarray(face_img)

        offset_x = landmarks.part(NOSE_TIP).x - scarf["points"]["33"][0]
        offset_y = landmarks.part(NOSE_TIP).y - scarf["points"]["33"][1]
        
        scarf_to_warp = Image.new('RGBA', (face_img.width, face_img.height))
        scarf_to_warp.paste(scarf["image"], (offset_x, offset_y), scarf["image"])

        current, projection = [], []
        for k, v in scarf["points"].items():
            current.append([
                v[0] + offset_x,
                v[1] + offset_y
            ])
            projection.append([
                landmarks.part(int(k)).x,
                landmarks.part(int(k)).y
            ])
        tform = tf.estimate_transform('affine', np.array(current), np.array(projection))
        warped_scarf = tf.warp(np.array(scarf_to_warp), tform.inverse, mode='edge', output_shape=(face_img.width, face_img.height), preserve_range=True)
        #warped_scarf = skimage.transform.warp(np.array(scarf_to_warp), tform.inverse, mode='edge', preserve_range=True)
        # warped_scarf = tf.matrix_transform(np.array(scarf_to_warp), tform)
        warped_scarf = warped_scarf.astype(np.uint8)
        warped_scarf = Image.fromarray(warped_scarf).convert("RGBA")
        face_img.paste(warped_scarf, (0, 0), warped_scarf)

        # face_img = np.array(face_img)
        # for c, p in zip(current, projection):
        #    face_img = cv2.circle(face_img, (c[0], c[1]), 6, (255, 0, 0), 5)
        #    face_img = cv2.circle(face_img, (p[0], p[1]), 5, (0, 255, 0), 5)

        return face_img


    def put_eyeglasses(self, face_img, landmarks, yaw):
        selected_pair = random.choice(self.glasses)
        glasses_left = selected_pair.crop((0, 0, selected_pair.width // 2, selected_pair.height))
        glasses_right = selected_pair.crop((selected_pair.width // 2, 0, selected_pair.width, selected_pair.height))
        face_img = Image.fromarray(face_img)

        # RESIZING
        height_l = int(landmarks_distance(landmarks.part(EYEBROW_TOP_L), landmarks.part(EYE_BOTTOM_L)))
        height_r = int(landmarks_distance(landmarks.part(EYEBROW_TOP_R), landmarks.part(EYE_BOTTOM_R)))
        new_height = (height_l + height_r)//2
        # LEFT PART RESIZING
        width_l = int(landmarks_distance(landmarks.part(EYEBROW_LIMIT_L), landmarks.part(NOSE_CENTER)))
        glasses_left = glasses_left.resize((width_l, new_height))
        # RIGHT PART RESIZING
        width_r = int(landmarks_distance(landmarks.part(EYEBROW_LIMIT_R), landmarks.part(NOSE_CENTER)))
        glasses_right = glasses_right.resize((width_r, new_height))

        # MERGING
        size = (glasses_left.width + glasses_right.width, new_height)
        glasses = Image.new('RGBA', size)
        if yaw < 45:
            glasses.paste(glasses_left, (0, 0), glasses_left)
        if yaw > -45:
            glasses.paste(glasses_right, (glasses_left.width, 0), glasses_right)

        offset = (width_l+width_r) // 2 - width_l
        offset_x = landmarks.part(NOSE_CENTER).x + int(offset) - glasses.width//2

        face_img.paste(glasses, (offset_x, landmarks.part(NOSE_CENTER).y-glasses.height//2), glasses)

        return face_img

    def put_hat(self, face_img, landmarks, face_width, face_x):
        selected_hat = random.choice(self.hats)
        hat_left = selected_hat.crop((0, 0, selected_hat.width // 2, selected_hat.height))
        hat_right = selected_hat.crop((selected_hat.width // 2, 0, selected_hat.width, selected_hat.height))
        face_img = Image.fromarray(face_img)

        # RESIZING
        width_left = landmarks.part(NOSE_CENTER).x - face_x
        width_right = face_x + face_width - landmarks.part(NOSE_CENTER).x
        new_height = int(selected_hat.height*(width_left+width_right)/selected_hat.width)
        hat_left = hat_left.resize((width_left, new_height))
        hat_right = hat_right.resize((width_right, new_height))
        size = (width_right + width_left, new_height)
        hat = Image.new('RGBA', size)
        hat.paste(hat_left, (0, 0), hat_left)
        hat.paste(hat_right, (width_left, 0), hat_right)

        offset_x = landmarks.part(NOSE_CENTER).x - width_left
        offset_y = max(landmarks.part(EYEBROW_TOP_L).y, landmarks.part(EYEBROW_TOP_R).y) - new_height

        face_img.paste(hat, (offset_x, offset_y), hat)

        return face_img

if __name__ == "__main__":
    styler = Styler("glasses", "hats", "scarves")

    img_path = "D:\\Documenti HDD\\GitHub\\AgeEstimationFramework\\dataset\\vggface2_data\\train\\n000004\\0151_01.jpg"

    rpy = RPYEstimator(os.path.join("models", "hopenet_robust_alpha1.pkl"))
    # estimated_angles = pickle.load(open("rpy_dlib.pkl", "rb"))
    img_color = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Face detection
    dets = detector(img_gray)
    det = correct_detection(dets[0], img_gray)
    ldmrks = predictor(img_gray, det)

    roll, pitch, yaw = rpy.estimate(det, img_color)

    axis_image = img_color.copy()
    draw_axis(axis_image, yaw, pitch, roll)
    cv2.imwrite("img_axes.png", axis_image)

    # Face rotation
    left_eye = extract_left_eye_center(ldmrks)
    right_eye = extract_right_eye_center(ldmrks)  
    dlib_angle = angle_between_2_points(left_eye, right_eye)
    if abs(roll.item()) > abs(dlib_angle):
        angle = roll.item()
    else:
        angle = dlib_angle
    M = cv2.getRotationMatrix2D((ldmrks.part(NOSE_CENTER).x, ldmrks.part(NOSE_CENTER).y), angle , 1)
    rotated = cv2.warpAffine(img_color, M, (img_color.shape[1], img_color.shape[0]), flags=cv2.INTER_CUBIC)

    cv2.imwrite("img_rotated.png", rotated)

    # Rotated landmarls
    ldmrks = predictor(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY), det)
    ldmrk_img = rotated.copy()

    for i in range(68):
        ldmrk_img = cv2.circle(ldmrk_img, (ldmrks.part(i).x, ldmrks.part(i).y), 3, (0, 255, 0), -1)
    cv2.imwrite("img_ldmrks.png", ldmrk_img)

    # Add glasses
    # face = styler.put_eyeglasses(rotated, ldmrks, yaw)
    # face = np.array(face)

    # Add hat
    # face = styler.put_hat(rotated, ldmrks, det.right()-det.left(), det.left())
    # face = np.array(face)

    # Add scarf
    # face = styler.put_scarf(rotated, ldmrks)
    # face = np.array(face)

    # cropped = crop_image(face, det)
    # save_path = os.path.join(output_subdir, img)
    # cv2.imwrite(save_path, cropped)

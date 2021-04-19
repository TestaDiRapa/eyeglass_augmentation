from PIL import Image, ImageFile
from utils import landmarks_distance, correct_detection, angle_between_2_points, extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image
import cv2
import dlib
import math
import numpy as np
import os
import pickle
import random
import sys


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib/shape_predictor_68_face_landmarks.dat")
EYEBROW_LIMIT_L = 17
EYEBROW_LIMIT_R= 26
EYE_LIMIT_L = 36
EYE_LIMIT_R = 45
NOSE_CENTER = 27
EYEBROW_TOP_L = 19
EYEBROW_TOP_R = 24
EYE_BOTTOM_L = 40
EYE_BOTTOM_R = 46

class Optician:
    
    def __init__(self, glasses_path):
        self.glasses = []
        for img in os.listdir(glasses_path):
            self.glasses.append(Image.open(os.path.join(glasses_path, img)))

    def put_eyeglasses(self, face_img, landmarks):
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
        glasses.paste(glasses_left, (0, 0), glasses_left)
        glasses.paste(glasses_right, (glasses_left.width, 0), glasses_right)

        # ROTATION
        # left_eye = extract_left_eye_center(landmarks)
        # right_eye = extract_right_eye_center(landmarks)
        left_eye = (landmarks.part(EYE_LIMIT_L).x, landmarks.part(EYE_LIMIT_L).y)
        right_eye = (landmarks.part(EYE_LIMIT_R).x, landmarks.part(EYE_LIMIT_R).y)
        angle = angle_between_2_points(left_eye, right_eye)
        glasses = glasses.rotate(angle, expand=True)
        radian = angle * np.pi / 18

        offset = (width_l+width_r) // 2 - width_l
        offset_x = landmarks.part(NOSE_CENTER).x + int(offset * np.cos(radian)) - glasses.width//2

        face_img.paste(glasses, (offset_x, landmarks.part(NOSE_CENTER).y-glasses.height//2), glasses)

        return face_img


if __name__ == "__main__":
    optician = Optician("glasses")


    # input_dir = "D:\\Documenti HDD\\GitHub\\AgeEstimationFramework\\dataset\\vggface2_data\\train"
    # output_dir = "train_eyeglasses"
    input_dir = sys.argv[1]
    output_dir = os.path.join("output", sys.argv[2])
    no_face = set()
    duplicate = set()

    for d in os.listdir(input_dir):
        output_subdir = os.path.join(output_dir, d)
        print(output_subdir)
        os.makedirs(output_subdir)
        dir_path = os.path.join(input_dir, d)
        for img in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img)
            img_color = cv2.imread(img_path)
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

            dets = detector(img_gray)
            if len(dets) == 0:
                no_face.add(img_path)
                pickle.dump(no_face, open(os.path.join("utils", "no_face_detected.pkl"), "wb"))
                
            for i, det in enumerate(dets):
                det = correct_detection(det, img_gray)
                ldmrks = predictor(img_gray, det)
                face = optician.put_eyeglasses(img_color, ldmrks)
                face = np.array(face)
                left_eye = extract_left_eye_center(ldmrks)
                right_eye = extract_right_eye_center(ldmrks)       
                M = get_rotation_matrix(left_eye, right_eye)
                rotated = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]), flags=cv2.INTER_CUBIC)
                cropped = crop_image(rotated, det)
                if i > 0:
                    save_path = os.path.join(output_subdir, img.split('.')[0], "_{}.jpg".format(i))
                    duplicate.add(save_path)
                    pickle.dump(duplicate, open(os.path.join("utils", "face_duplicate.pkl"), "wb"))
                else:
                    save_path = os.path.join(output_subdir, img)
                cv2.imwrite(save_path, cropped)
                # cv2.imshow("Prova", cropped)
                # cv2.waitKey(-1)
                # cv2.destroyAllWindows()
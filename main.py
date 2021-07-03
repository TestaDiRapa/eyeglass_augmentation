from obstruction_augmentation import Styler, RPYEstimator
from tqdm import tqdm
from utils import correct_detection, angle_between_2_points, extract_left_eye_center, extract_right_eye_center, crop_image
import argparse
import cv2
import dlib
import numpy as np
import os
import pickle
import sys

NOSE_CENTER = 27
DETECTOR_CHOICE = ["dlib", "csv"]
MODE_CHOICE = ["augment", "meta"]
RPY_CHOICE = ["hopenet", "dump"]

class RPYDump:

    def __init__(self, filename):
        print("Loading RPY dump file")
        self.dump = pickle.load(open(filename, "rb"))
    
    def __getitem__(self, index):
        new_index = "D:\\Documenti HDD\\GitHub\\AgeEstimationFramework\\dataset\\vggface2_data\\train\\" + index
        return self.dump[new_index]

class CSVMeta:

    def __init__(self, filename):
        print("Loading meta file")
        self.data = dict()
        with open(filename, "r") as infile:
            for line in tqdm(infile.readlines()):
                info = line[:-1].split(',')
                left = int(info[4])
                top = int(info[5])
                right = left + int(info[6])
                bottom = top + int(info[7])
                self.data[info[2]] = dlib.rectangle(left, top, right, bottom)

    def __getitem__(self, index):
        if index in self.data:
            return [self.data[index]]
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script settings.')

    parser.add_argument('--base_folder', type=str, default="D:\\Documenti HDD\\GitHub\\AgeEstimationFramework\\dataset\\vggface2_data", help='Base folder')
    parser.add_argument('--face_detector', type=str, default='dlib', choices=DETECTOR_CHOICE)
    parser.add_argument('--glasses', type=bool, default=False, help='Wheter to apply eyeglasses')
    parser.add_argument('--mask', type=bool, default=False, help='Wheter to apply the mask')
    parser.add_argument('--meta_filename', type=str, default="meta.csv", help="Meta filename")
    parser.add_argument('--mode', type=str, default='augment', choices=MODE_CHOICE)
    parser.add_argument('--output_folder', type=str, default=None, help='Output folder')
    parser.add_argument('--partition', type=str, default="test", help='Train/test/validation')
    parser.add_argument('--rpy_estimation', type=str, default='hopenet', choices=RPY_CHOICE)
    parser.add_argument('--save_no_face', type=bool, default=False, help='Wheter to save the images where no face is detected')
    parser.add_argument('--scarf', type=bool, default=False, help='Wheter to apply the scarf')

    args = parser.parse_args()

    predictor = dlib.shape_predictor("dlib/shape_predictor_68_face_landmarks.dat")
    styler = Styler("glasses", "scarves")

    input_dir = os.path.join(args.base_folder, args.partition)
    output_dir = os.path.join("output", args.output_folder)
    
    no_face = set()
    meta_buffer = list()
    meta_counter = 0

    if args.rpy_estimation == "hopenet":
        rpy = RPYEstimator(os.path.join("models", "hopenet_robust_alpha1.pkl"))
    elif args.rpy_estimation == "dump":
        rpy = RPYDump("rpy_dlib.pkl")

    if args.face_detector == "dlib":
        detector = dlib.get_frontal_face_detector()
    elif args.face_detector == "csv":
        detector = CSVMeta("train.detected.csv")

    for d in tqdm(os.listdir(input_dir)):
        output_subdir = os.path.join(output_dir, d)
        if not os.path.isdir(output_subdir):
            os.makedirs(output_subdir)
            dir_path = os.path.join(input_dir, d)

            for img in os.listdir(dir_path):
                img_path = os.path.join(dir_path, img)
                img_color = cv2.imread(img_path)
                img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

                # Face detection
                if args.face_detector == "dlib":
                    dets = detector(img_gray)
                elif args.face_detector == "csv":
                    dets = detector['{}/{}'.format(d, img)]
                
                if len(dets) == 0 and args.save_no_face:
                    no_face.add(img_path)
                    pickle.dump(no_face, open(os.path.join("utils", "{}_no_face_detected.pkl".format(sys.argv[2])), "wb"))
                elif len(dets) > 0:
                    det = correct_detection(dets[0], img_gray)
                    ldmrks = predictor(img_gray, det)

                    # Face rotation
                    if args.rpy_estimation == "hopenet":
                        roll, pitch, yaw = rpy.estimate(det, img_color)
                    elif args.rpy_estimation == "dump":
                        roll, pitch, yaw = rpy["{}\\{}".format(d, img)]

                    left_eye = extract_left_eye_center(ldmrks)
                    right_eye = extract_right_eye_center(ldmrks)  
                    dlib_angle = angle_between_2_points(left_eye, right_eye)
                    if abs(roll) > abs(dlib_angle):
                        angle = roll
                    else:
                        angle = dlib_angle
                    M = cv2.getRotationMatrix2D((ldmrks.part(NOSE_CENTER).x, ldmrks.part(NOSE_CENTER).y), angle , 1)
                    rotated = cv2.warpAffine(img_color, M, (img_color.shape[1], img_color.shape[0]), flags=cv2.INTER_CUBIC)

                    if args.mode == "augment":
                        # Rotated landmarls
                        ldmrks = predictor(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY), det)

                        if args.glasses:
                            face = styler.put_eyeglasses(rotated, ldmrks, yaw)
                            face = np.array(face)

                        if args.scarf:
                            face = styler.put_scarf(rotated, ldmrks)
                            face = np.array(face)

                        cropped = crop_image(face, det)
                        save_path = os.path.join(output_subdir, img)
                        cv2.imwrite(save_path, cropped)

                    elif args.mode == "meta":
                        new_ldmrks = predictor(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY), det)
                        formatter = {
                            "folder": d,
                            "image": img,
                            "left": det.left(),
                            "top": det.top(),
                            "width": det.right()-det.left(),
                            "height": det.bottom()-det.top(),
                            "roll": roll,
                            "yaw": yaw,
                            "center_x": ldmrks.part(NOSE_CENTER).x,
                            "center_y": ldmrks.part(NOSE_CENTER).y,
                        }
                        for i in [2, 8, 14, 17, 19, 24, 26, 27, 33, 40, 46]:
                            formatter["l{}_x".format(i)] = ldmrks.part(i).x
                            formatter["l{}_y".format(i)] = ldmrks.part(i).y
                        line = "0,0,{folder}/{image},{folder},{left},{top},{width},{height},{roll:.3f},{yaw:.3f},{center_x},{center_y},"\
                        "{l2_x},{l2_y},{l8_x},{l8_y},{l14_x},{l14_y},{l17_x},{l17_y},{l19_x},{l19_y},{l24_x},{l24_y},{l26_x},{l26_y},{l27_x},{l27_y},"\
                        "{l33_x},{l33_y},{l40_x},{l40_y},{l46_x},{l46_y}\n".format(**formatter)
                        meta_buffer.append(line)
                        meta_counter +=1
                        if meta_counter == 100:
                            meta_counter = 0
                            with open(args.meta_filename, 'a') as outfile:
                                for line in meta_buffer:
                                    outfile.write(line)
                            meta_buffer = []
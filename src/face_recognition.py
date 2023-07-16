# import cv2
# from PIL import Image
# import argparse
# from pathlib import Path
# from multiprocessing import Process, Pipe, Value, Array
# import torch
# from config import get_config
# from mtcnn import MTCNN
# from Learner import face_learner
# from utils import load_facebank, draw_box_name, prepare_facebank
# from deepface import DeepFace
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='for face verification')
#     parser.add_argument("-s", "--save", help="whether save", action="store_true")
#     parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
#     parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")
#     parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
#     parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
#     args = parser.parse_args()
#
#     conf = get_config(False)
#     conf.network = 'vit'
#
#     learner = face_learner(conf, True)
#     learner.threshold = args.threshold
#
#     learner.model.eval()
#     print('learner loaded')
#
#
#     if args.update:
#         targets, names = prepare_facebank(conf, learner.model, mtcnn, tta=args.tta)
#         print('facebank updated')
#     else:
#         targets, names = load_facebank(conf)
#         print('facebank loaded')
#
#     cap = cv2.VideoCapture(0)
#     cap.set(3, 1280)
#     cap.set(4, 720)
#
#     while cap.isOpened():
#         isSuccess, frame = cap.read()
#         if isSuccess:
#             try:
#                 face_objs = DeepFace.extract_faces(img_path="img.jpg",
#                                                    target_size=(112, 112),
#                                                    detector_backend='yolov8')
#             except:
#                 print('detect error')
#
#             cv2.imshow('face Capture', frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()

from deepface import DeepFace

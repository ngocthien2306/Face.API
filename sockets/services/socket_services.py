import cv2
from PIL import Image
import numpy as np
from src.config import get_config
from src.Learner import face_learner
from src.database.connect import connection
from src.mtcnn import MTCNN
from src.services.face.face_services import FaceCheckInService
from src.utils import draw_box_name, load_facebank_csv, assign_face_bank_all, play_sound
import traceback
import time
from src.detect.src.align_trans import get_reference_facial_points
reference = get_reference_facial_points(default_square=True)


class FaceServices:
    _instance = None

    # def __new__(cls, *args, **kwargs):
    #     if not cls._instance:
    #         cls._instance = super(FaceServices, cls).__new__(cls, *args, **kwargs)
    #     return cls._instance

    def __init__(self, threshold=1.4, network='r34', tta=True, update=False, pkl=True, show_score=True):
        self.is_real_time_running = None
        self.conf = get_config(False)
        self.conf.network = network
        self.curr_network = network
        self.pkl = pkl
        self.show_score = show_score
        self.tta = tta
        self.mtcnn = MTCNN()
        self.learner = face_learner(self.conf, True)
        self.threshold = threshold
        self.learner.threshold = self.threshold
        if self.conf.network in ['ir_se50', 'mobilefacenet']:
            self.learner.load_state(self.conf, str(self.conf.network) + '.pth', True, True)
        self.learner.model.eval()
        self.update = update
        self.targets = None
        self.names = None
        self.representations = None
        self.get_datasource()
        self.attempt = 40
        self.curr_user = None

    def change_model(self):
        self.learner = face_learner(self.conf, True)
        if self.conf.network in ['ir_se50', 'mobilefacenet']:
            self.learner.load_state(self.conf, str(self.conf.network) + '.pth', True, True)
        self.learner.model.eval()

    def get_datasource(self):
        if self.update:
            self.targets, self.names, self.representations = assign_face_bank_all(self.conf, self.learner.model,
                                                                                  self.mtcnn, tta=self.tta)
        else:
            self.targets, self.names, self.representations = load_facebank_csv(self.conf)

    def get_db_path(self):
        paths = []
        for idx, main_path in enumerate([self.conf.no_user_path, self.conf.user_path]):
            for path in main_path.iterdir():
                paths.append(path)
        return paths

    def recognize_user(self, image):
        try:
            image = Image.fromarray(image)
            bboxes, faces = self.mtcnn.align_multi(image, self.conf.face_limit, self.conf.min_face_size)
            bboxes = bboxes[:, :-1]
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1, -1, 1, 1]
            results = self.learner.infer_csv(self.conf, faces, self.representations, True)
            return {
                'bboxes': bboxes.tolist(),
                'results': results,
                'face_exist': True
            }
        except:
            traceback.print_exc()
            return {'bboxes': [], 'results': [], 'face_exist': False}

    def real_time_check_in(self):
        self.is_real_time_running = True  # Add a flag to track the running state
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)
        fps_interval = 1  # Number of seconds between FPS updates
        fps_start_time = time.time()
        fps_frame_count = 0
        cv2.namedWindow('face Capture', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('face Capture', 1000, 562)

        while cap.isOpened() and self.is_real_time_running:  # Check the flag to continue or stop the process
            isSuccess, frame = cap.read()
            if isSuccess:
                try:
                    image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
                    # image = Image.fromarray(frame)

                    bboxes, faces = self.mtcnn.align_multi(image, self.conf.face_limit, self.conf.min_face_size)

                    bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibility faces
                    bboxes = bboxes.astype(int)
                    bboxes = bboxes + [-1, -1, 1, 1]  # personal choice

                    results = self.learner.infer_csv(self.conf, faces, self.representations, self.tta)

                    for idx, bbox in enumerate(bboxes):
                        if self.show_score:
                            frame = draw_box_name(bbox, results[0]['identity'][idx] + '_{:.2f}'.format(
                                results[0]['distances'][0]), frame)
                        else:
                            frame = draw_box_name(bbox, results[idx][0], frame)

                        if self.curr_user != results[0]['identity'][idx] or results[0]['identity'][idx] == 'Unknown':
                            self.curr_user = results[0]['identity'][idx]
                            self.attempt = 40
                        else:
                            self.attempt -= 1

                        if self.attempt == 0:
                            connection.connect()
                            service = FaceCheckInService(connection)
                            data = {
                                'userId': self.curr_user,
                                'approvalType': 'APPTP5',
                                'faceImage': faces[idx]
                            }
                            try:
                                service.insert_record(data)
                                wav_file = "./public/files/audio/checkin.wav"
                                play_sound(wav_file)

                            except:
                                connection.disconnect()

                            connection.disconnect()
                            self.attempt = 40


                except Exception as e:
                    traceback.print_exc()
                    pass

                cv2.imshow('face Capture', frame)

                # Calculate FPS
                fps_frame_count += 1
                if time.time() - fps_start_time >= fps_interval:
                    fps = fps_frame_count / (time.time() - fps_start_time)
                    # print("FPS:", round(fps, 2))
                    fps_frame_count = 0
                    fps_start_time = time.time()

                if cap.get(cv2.CAP_PROP_POS_MSEC) % 300 == 0:
                    if len(self.get_db_path()) != len(self.representations):
                        self.get_datasource()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def stop_real_time_check_in(self):
        self.is_real_time_running = False  # Set the flag to stop the process

    def convertbase64(self, string64):
        np_data = np.fromstring(string64, np.uint8)
        image = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        return image

import cv2
from PIL import Image
import argparse
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank, assign_facebank, load_facebank_csv
import time
import traceback

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save", action="store_true")
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.3, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
    parser.add_argument('-net', '--network', help="network", default='vit')
    args = parser.parse_args()

    conf = get_config(False)
    conf.network = args.network

    mtcnn = MTCNN()
    print('mtcnn loaded')

    learner = face_learner(conf, True)
    learner.threshold = args.threshold

    if conf.network in ['ir_se50', 'mobilefacenet']:
        learner.load_state(conf, str(conf.network) + '.pth', True, True)

    learner.model.eval()
    print('learner loaded')

    if args.update:
        targets, names, representations = assign_facebank(conf, learner.model, mtcnn, tta=args.tta)
        print(names)
        print(representations[0])
        print('facebank updated')
    else:
        targets, names, representations = load_facebank_csv(conf)
        print(len(representations))
        print('facebank loaded')


    # inital camera
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    # if args.save:
    #     video_writer = cv2.VideoWriter(conf.data_path / 'recording.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 6, (1280, 720))
        # frame rate 6 due to my laptop is quite slow...
    fps_interval = 1  # Number of seconds between FPS updates
    fps_start_time = time.time()
    fps_frame_count = 0

    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:
            try:
                # image = Image.fromarray(frame[...,::-1]) #bgr to rgb
                image = Image.fromarray(frame)
                bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibility faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1, -1, 1, 1]  # personal choice
                results = learner.infer_csv(conf, faces, representations, args.tta)
                for idx, bbox in enumerate(bboxes):
                    if args.score:
                        frame = draw_box_name(bbox, results[idx][0] + '_{:.2f}'.format(results[idx][2]), frame)
                    else:
                        frame = draw_box_name(bbox, results[idx][0], frame)
                print(results)
            except Exception as e:
                continue
                # print("Detect Error")
                # traceback.print_exc()

            cv2.imshow('face Capture', frame)

            # Calculate FPS
            fps_frame_count += 1
            if time.time() - fps_start_time >= fps_interval:
                fps = fps_frame_count / (time.time() - fps_start_time)
                print("FPS:", round(fps, 2))
                fps_frame_count = 0
                fps_start_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
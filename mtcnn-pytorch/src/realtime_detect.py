import cv2
from PIL import Image
import time
from src import detect_faces, show_bboxes

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
            image = Image.fromarray(frame)
            bounding_boxes, landmarks = detect_faces(image, thresholds=[0.6, 0.7, 0.85])

            for idx, bbox in enumerate(bounding_boxes):
                frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
                frame = cv2.putText(frame,
                                'thien',
                                (bbox[0],bbox[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                2,
                                (0,255,0),
                                3,
                                cv2.LINE_AA)

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
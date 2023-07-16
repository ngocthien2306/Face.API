from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import numpy as np
from app.user.enums.user import UserEnum
from app.user.helper.ocr.text_recognition import TextRecognition

class yoloDetect:
    def __init__(self):
        self.detect_path = "/data/thinhlv/hung/Capstone/cccd_detect_character/runs/detect/train39/weights/best.pt"
        self.text_recognition_model = TextRecognition()
        self.model = YOLO(str(self.detect_path)) 
    def detect_id(self, image) -> str:
        try:
            
            image = Image.fromarray(image)
            
            results = self.model(image, iou=0.1)
            x1, y1, x2, y2,acc,l = None, None, None, None, None, None
            x1_title, y1_title, x2_title, y2_title,acc_title,l_title = None, None, None, None, None, None
            if len(results[0].boxes.data) == UserEnum.NUMBER_FIELD_CCCD.value:
                for data in results[0].boxes.data:
                    result = data.to('cpu').tolist()
                    if(result[5] == UserEnum.LABLE_ID.value): 
                        x1, y1, x2, y2,acc,l = result
                    elif(result[5] == UserEnum.LABLE_TITLE.value):
                        x1_title, y1_title, x2_title, y2_title,acc_title,l_title = result
                if (x1_title== None or y1_title== None or x2_title== None or y2_title== None or acc_title== None or l_title == None):
                    return UserEnum.STATUS_CCCD_4.value
                elif(x1 == None or y1 == None or x2 == None or y2 == None or acc == None or l == None):
                    return UserEnum.STATUS_CCCD_3.value
                else:
                    image_croped = image
                    image_croped_title = image
                    
                    ## Verify Title
                    cropped_image_title = image_croped_title.crop((x1_title, y1_title, x2_title, y2_title))
                    cv2_image_title = cv2.cvtColor(np.array(cropped_image_title), cv2.COLOR_RGB2BGR)
                    cv2.imwrite("/data/thinhlv/hung/Capstone/cccd_detect_character/testTitle.jpg", cv2_image_title)
                    title = self.text_recognition_model.predict(cv2_image_title)
                    print("title",title)
                    cropped_image = image_croped.crop((x1, y1, x2, y2))
                    cv2_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
                    id = self.text_recognition_model.predict(cv2_image)
                    print(id)
                    if (len(id) == UserEnum.NUMBER_ID_CCCD.value):
                        return id
                    return UserEnum.STATUS_CCCD_4.value
            else:
                return UserEnum.STATUS_CCCD_4.value
        except Exception as e:
            print(e)
            return str(e)
    def detect_idImage(self, image) -> str:
        try:
            image = Image.fromarray(image)
            image.save("/data/thinhlv/hung/Capstone/ML.API/testImage2.png")
            cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.imwrite("/data/thinhlv/hung/Capstone/ML.API/testImage.jpg", cv2_image)
            id = self.text_recognition_model.predict(cv2_image)
            print(id)
            return id
        except Exception as e:
            print(e)
            return str(e)

    
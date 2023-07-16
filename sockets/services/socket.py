from PIL import Image
import cv2
import torch
import math 
import sockets.function.utils_rotate as utils_rotate
from IPython.display import display
import os
import sockets.function.helper as helper

import math
yolo_LP_detect = torch.hub.load('ultralytics/yolov5', 'custom', path='sockets/models/bestlp640.pt', force_reload=False )
yolo_license_plate = torch.hub.load('ultralytics/yolov5', 'custom', path='sockets/models/best_orc.pt', force_reload=False)

yolo_license_plate.conf = 0.60

listlb = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','K','L','M', 'N','P','S','T','U','V','X','Y','Z','0']
# license plate type classification helper function
def linear_equation(x1, y1, x2, y2):
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b

def check_point_linear(x, y, x1, y1, x2, y2):
    a, b = linear_equation(x1, y1, x2, y2)
    y_pred = a*x+b
    return(math.isclose(y_pred, y, abs_tol = 3))
def read_plate(yolo_license_plate, im):
    LP_type = "1"
    results = yolo_license_plate(im)
    bb_list = results.pandas().xyxy[0].values.tolist()

    #if len(bb_list) == 0 or len(bb_list) < 7 or len(bb_list) > 10:
    if len(bb_list) == 0:
        return "unknown"
    center_list = []
    y_mean = 0
    y_sum = 0
    for bb in bb_list:
        x_c = (bb[0]+bb[2])/2
        y_c = (bb[1]+bb[3])/2
        y_sum += y_c
        center_list.append([x_c,y_c,bb[-1]])

    # find 2 point to draw line
    l_point = center_list[0]
    r_point = center_list[0]
    for cp in center_list:
        if cp[0] < l_point[0]:
            l_point = cp
        if cp[0] > r_point[0]:
            r_point = cp
    for ct in center_list:
        if l_point[0] != r_point[0]:
            if (check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]) == False):
                LP_type = "2"

    y_mean = int(int(y_sum) / len(bb_list))
    size = results.pandas().s

    # 1 line plates and 2 line plates
    line_1 = []
    line_2 = []
    license_plate = ""
    if LP_type == "2":
        for c in center_list:
            if int(c[1]) > y_mean:
                line_2.append(c)
            else:
                line_1.append(c)
        for l1 in sorted(line_1, key = lambda x: x[0]):
            license_plate += str(listlb[int(l1[2])])
        license_plate += "-"
        for l2 in sorted(line_2, key = lambda x: x[0]):
            license_plate += str(listlb[int(l2[2])])
    else:
        for l in sorted(center_list, key = lambda x: x[0]):
            license_plate += str(listlb[int(l[2])])
    return license_plate
def LP_detect(image):
    plates = yolo_LP_detect(image, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()
    count = 0
    if len(list_plates) == 0:
        lp = helper.read_plate(yolo_license_plate,image)
        if lp != "unknown":
            list_read_plates.add(lp)
    else:
        for plate in list_plates:
            flag = 0
            x = int(plate[0]) # xmin
            y = int(plate[1]) # ymin
            w = int(plate[2] - plate[0]) # xmax - xmin
            h = int(plate[3] - plate[1]) # ymax - ymin  
            crop_img = image[y:y+h, x:x+w]
            cv2.rectangle(image, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
            # cv2.imwrite("crop.jpg", crop_img)
            # rc_image = cv2.imread("crop.jpg")
            lp = ""
            count+=1
            for cc in range(0,2):
                for ct in range(0,2):
                    # Check Rotate of LP
                    lp = read_plate(yolo_license_plate, crop_img)
                    cv2.imwrite("public/temp/cropRotate.jpg", utils_rotate.deskew(crop_img, cc, ct))
                    if lp != "unknown":
                        print(lp)
                        list_read_plates.add(lp)
                        flag = 1
                        break
                if flag == 1:
                    break
    return list_read_plates
    
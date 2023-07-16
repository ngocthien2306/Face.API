import torch
import os
import numpy as np
from sklearn import preprocessing
import cv2
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from app.track.helper.embedding import euclid_distance
class FaceServices:

    def embedding_face(self, model, img):
        #img = load_img(employee, target_size=(112, 112))
        img = img_to_array(img)
        img_flip = cv2.flip(img, 1)
        img_c = np.array([img, img_flip])
        img_c = np.transpose(img_c, (0, 3, 1, 2))
        img_1 = ((img_c[:1] / 255) - 0.5) / 0.5
        img_2 = ((img_c[1:2] / 255) - 0.5) / 0.5
        net_out_1 = model.cuda(0)(torch.from_numpy(img_1))
        embedding1 = net_out_1.detach().cpu().numpy()
        net_out_2 = model.cuda(0)(torch.from_numpy(img_2))
        embedding2 = net_out_2.detach().cpu().numpy()
        img_representation = preprocessing.normalize(embedding1 + embedding2) 
        
        return img_representation 
    def face_check_track(self,model, img_verify, img_db_path, threshor=1.4) -> bool:
        try:
            e_verify = self.embedding_face(model, img_verify)
            img_db = load_img(img_db_path, target_size=(112, 112))
            e_db = self.embedding_face(model, img_db)
            dist = euclid_distance(e_db, e_verify)
            print("Accuracy",dist)
            return  True if dist < threshor else False
        except Exception as e:
            print(e)
            return False
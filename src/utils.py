from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import io
from torchvision import transforms as trans
from src.data.data_pipe import de_preprocess
import torch
from src.model import l2_norm
import cv2
import pickle
import pygame
from pydub import AudioSegment
from datetime import datetime

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

shifts = [['05:00:00', '10:59:59', './public/files/audio/morning.wav'],
          ['11:00:00', '13:29:59', './public/files/audio/lunch.wav'],
          ['13:30:00', '17:59:59', './public/files/audio/after.wav'],
          ['18:00:00', '21:59:59', './public/files/audio/evening.wav'],
          ['22:00:00', '04:59:59', './public/files/audio/evening.wav']]


def convert_to_wav(mp3_file, wav_file):
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format="wav")

def play_sound(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

def schedule_play_audio():
    time_now = str(datetime.now().time())
    file = ''
    for shift in shifts:
        if shift[0] < time_now < shift[1]:
            file = shift[2]
    return file

def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


def embedding_face(img, model):
    # Flip the image horizontally
    img_flip = img.transpose(method=Image.FLIP_LEFT_RIGHT)

    # Prepare transformations for normalization
    transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Apply transformations to the original and flipped images
    img_tensor = transform(img).unsqueeze(0)
    img_flip_tensor = transform(img_flip).unsqueeze(0)

    # Send the tensors to the GPU (assuming CUDA is available)
    img_tensor = img_tensor.to(device)
    img_flip_tensor = img_flip_tensor.to(device)

    # Pass the tensors through the model

    embedding1 = model(img_tensor)
    embedding2 = model(img_flip_tensor)

    # Normalize the embeddings
    embeddings_sum = embedding1 + embedding2
    normalized_embedding = torch.nn.functional.normalize(embeddings_sum)

    return normalized_embedding

def assign_face_bank_all(conf, model, mtcnn, tta=True):
    print("assign_facebank")
    model.eval()
    embeddings = []
    representations = []

    names = ['Unknown']
    for idx, main_path in enumerate([conf.user_path]):
        for path in main_path.iterdir():
            path_new = path
            folder_name = ''

            if path.is_file():
                continue
            else:
                emb_by_user = []
                path = path / 'face'
                try:
                    for file in path.iterdir():
                        if not file.is_file():
                            continue
                        else:
                            try:
                                img = Image.open(file)

                            except:
                                continue
                            if img.size != (112, 112):
                                try:
                                    img = mtcnn.align(img)
                                except:
                                    continue

                            if idx == 1:
                                folder_name = path_new.name
                            else:
                                folder_name = path.name

                            with torch.no_grad():
                                if tta:
                                    if conf.network in ['r100', 'vit', 'r34', 'r18', 'mbf']:
                                        emb = embedding_face(img, model)
                                        emb_by_user.append(emb)
                                        # representations.append([folder_name, emb.cpu().detach().numpy(), folder])
                                    else:
                                        flip_img = trans.functional.hflip(img)
                                        emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                                        emb_mirror = model(conf.test_transform(flip_img).to(conf.device).unsqueeze(0))
                                        emb_by_user.append(l2_norm(emb + emb_mirror))
                                else:
                                    emb_by_user.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
                except:
                    continue
            if len(emb_by_user) == 0:
                continue
            embedding = torch.cat(emb_by_user).mean(0, keepdim=True)
            embeddings.append(embedding)
            representations.append([folder_name, embedding.cpu().detach().numpy(), path])
            names.append(path.name)

    embeddings = torch.cat(embeddings)
    names = np.array(names)

    with open(conf.facebank_path / '{0}_facebank_csv.pkl'.format(str(conf.network)), "wb") as f:
        print("FaceServices: saving...")
        pickle.dump(representations, f)
    torch.save(embeddings, conf.facebank_path / '{0}_facebank_csv.pth'.format(str(conf.network)))
    np.save(conf.facebank_path / '{0}_names_csv'.format(str(conf.network)), names)
    return embeddings, names, representations

def assign_facebank(conf, model, mtcnn, tta=True):
    print("assign_facebank")
    model.eval()
    embeddings = []
    representations = []
    names = ['Unknown']

    for path in conf.facebank_path.iterdir():
        if path.is_file():
            continue
        else:
            emb_by_user = []
            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    try:
                        img = Image.open(file)
                    except:
                        continue
                    if img.size != (112, 112):
                        try:
                            img = mtcnn.align(img)
                        except:
                            continue
                    with torch.no_grad():
                        if tta:
                            if conf.network in ['r100', 'vit', 'r34', 'r18', 'mbf']:
                                emb = embedding_face(img, model)
                                emb_by_user.append(emb)
                                representations.append([path.name, emb.cpu().detach().numpy(), 'test'])
                            else:
                                flip_img = trans.functional.hflip(img)
                                emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                                emb_mirror = model(conf.test_transform(flip_img).to(conf.device).unsqueeze(0))
                                emb_by_user.append(l2_norm(emb + emb_mirror))
                        else:
                            emb_by_user.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))

        if len(emb_by_user) == 0:
            continue

        embedding = torch.cat(emb_by_user).mean(0, keepdim=True)
        embeddings.append(embedding)
        names.append(path.name)

    embeddings = torch.cat(embeddings)
    names = np.array(names)
    # df = pd.DataFrame(representations, columns=["identity", f"{conf.network}_representation", "plate_num"])
    with open(conf.facebank_path / '{0}_facebank_csv.pkl'.format(str(conf.network)), "wb") as f:
        print("FaceServices: saving...")
        pickle.dump(representations, f)
    torch.save(embeddings, conf.facebank_path / '{0}_facebank_csv.pth'.format(str(conf.network)))
    np.save(conf.facebank_path / '{0}_names_csv'.format(str(conf.network)), names)
    return embeddings, names, representations
def prepare_facebank(conf, model, mtcnn, tta=True):
    model.eval()
    embeddings = []
    names = ['Unknown']
    for path in conf.facebank_path.iterdir():
        if path.is_file():
            continue
        else:
            embs = []
            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    try:
                        img = Image.open(file)
                    except:
                        continue
                    if img.size != (112, 112):
                        try:
                            img = mtcnn.align(img)
                        except:
                            continue
                    with torch.no_grad():
                        if tta:
                            mirror = trans.functional.hflip(img)
                            emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                            emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                            embs.append(l2_norm(emb + emb_mirror))
                        else:
                            embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        if len(embs) == 0:
            continue
        embedding = torch.cat(embs).mean(0, keepdim=True)
        embeddings.append(embedding)
        names.append(path.name)

    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, conf.facebank_path / '{0}_facebank.pth'.format(str(conf.network)))
    np.save(conf.facebank_path / '{0}_names'.format(str(conf.network)), names)
    return embeddings, names


def load_facebank(conf):
    embeddings = torch.load(conf.facebank_path / '{0}_facebank.pth'.format(str(conf.network)))
    names = np.load(conf.facebank_path / '{0}_names.npy'.format(str(conf.network)))
    return embeddings, names


def load_facebank_csv(conf):
    embeddings = torch.load(conf.facebank_path / '{0}_facebank_csv.pth'.format(str(conf.network)))
    names = np.load(conf.facebank_path / '{0}_names_csv.npy'.format(str(conf.network)))
    with open(conf.facebank_path / '{0}_facebank_csv.pkl'.format(str(conf.network)), "rb") as f:
        representations = pickle.load(f)
    return embeddings, names, representations


def face_reader(conf, conn, flag, boxes_arr, result_arr, learner, mtcnn, targets, tta):
    while True:
        try:
            image = conn.recv()
        except:
            continue
        try:
            bboxes, faces = mtcnn.align_multi(image, limit=conf.face_limit)
        except:
            bboxes = []

        results = learner.infer(conf, faces, targets, tta)

        if len(bboxes) > 0:
            print('bboxes in reader : {}'.format(bboxes))
            bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1, -1, 1, 1]  # personal choice
            assert bboxes.shape[0] == results.shape[0], 'bbox and faces number not same'
            bboxes = bboxes.reshape([-1])
            for i in range(len(boxes_arr)):
                if i < len(bboxes):
                    boxes_arr[i] = bboxes[i]
                else:
                    boxes_arr[i] = 0
            for i in range(len(result_arr)):
                if i < len(results):
                    result_arr[i] = results[i]
                else:
                    result_arr[i] = -1
        else:
            for i in range(len(boxes_arr)):
                boxes_arr[i] = 0  # by default,it's all 0
            for i in range(len(result_arr)):
                result_arr[i] = -1  # by default,it's all -1
        print('boxes_arr ： {}'.format(boxes_arr[:4]))
        print('result_arr ： {}'.format(result_arr[:4]))
        flag.value = 0


hflip = trans.Compose([
    de_preprocess,
    trans.ToPILImage(),
    trans.functional.hflip,
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf


def draw_box_name(bbox, name, frame):
    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
    frame = cv2.putText(frame,
                        name,
                        (bbox[0], bbox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,  # Adjust the font scale value to make the text smaller
                        (0, 255, 0),
                        1,  # Adjust the thickness of the text
                        cv2.LINE_AA)
    return frame

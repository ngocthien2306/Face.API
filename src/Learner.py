from src.data.data_pipe import de_preprocess, get_train_loader, get_val_data
from src.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from src.verifacation import evaluate
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import pandas as pd
import sklearn
plt.switch_backend('agg')
from src.utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import math
from src.backbones import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"
def load_model(weight_path, network):
    weight = torch.load(weight_path, map_location=torch.device(device))

    model = get_model(network, dropout=0, fp16=True).to(device)
    model.load_state_dict(weight)
    model = torch.nn.DataParallel(model)
    model.eval()
    return model


class face_learner(object):
    def __init__(self, conf, inference=False):
        if conf.network == 'mobilefacenet':
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')
        elif conf.network == 'vit':
            self.model = load_model('src/work_spaces/save/model_vit_s.pt', 'vit_s')
            print("Vit model generated")
        elif conf.network == 'r100':
            self.model = load_model('src/work_spaces/save/model_r100.pth', 'r100')
            print("R100 model generated")
        elif conf.network == 'r34':
            self.model = load_model('src/work_spaces/save/r34.pth', 'r34')
            print("R34 model generated")
        elif conf.network == 'r18':
            self.model = load_model('src/work_spaces/save/r18.pth', 'r18')
            print("R18 model generated")
        elif conf.network == 'mbf':
            self.model = load_model('src/work_spaces/save/model_mbf.pt', 'mbf')
            print("MB model generated")
        elif conf.network == 'ir_se50':
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        else:
            self.model = load_model('work_spaces/save/model_vit_s.pt', 'vit_s')
            print("Vit default model generated")

        if not inference:
            self.milestones = conf.milestones
            self.loader, self.class_num = get_train_loader(conf)

            self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)

            print('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                    {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr, momentum=conf.momentum)
            else:
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr, momentum=conf.momentum)
            print(self.optimizer)
            # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)

            print('optimizers generated')
            self.board_loss_every = len(self.loader) // 100
            self.evaluate_every = len(self.loader) // 10
            self.save_every = len(self.loader) // 5
            self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(
                self.loader.dataset.root.parent)
        else:
            self.threshold = conf.threshold

    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        torch.save(
            self.model.state_dict(), save_path /
                                     ('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step,
                                                                                   extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                                        ('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step,
                                                                                     extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                                             ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy,
                                                                                               self.step, extra)))

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.load_state_dict(torch.load(save_path / 'model_{}'.format(fixed_str), map_location=device))

        if not model_only:
            self.head.load_state_dict(torch.load(save_path / 'head_{}'.format(fixed_str), map_location=device))
            self.optimizer.load_state_dict(
                torch.load(save_path / 'optimizer_{}'.format(fixed_str), map_location=device))

    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)

    #         self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
    #         self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
    #         self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)

    def evaluate(self, conf, carray, issame, nrof_folds=5, tta=False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch.cpu())
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.to(conf.device)).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(conf.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

    def find_lr(self,
                conf,
                init_value=1e-8,
                final_value=10.,
                beta=0.98,
                bloding_scale=3.,
                num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):

            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)
            batch_num += 1

            self.optimizer.zero_grad()

            embeddings = self.model(imgs)
            thetas = self.head(embeddings, labels)
            loss = conf.ce_loss(thetas, labels)

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss, batch_num)
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
            # Do the SGD step
            # Update the lr for the next step

            loss.backward()
            self.optimizer.step()

            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses

    def train(self, conf, epochs):
        self.model.train()
        running_loss = 0.
        for e in range(epochs):
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()
            if e == self.milestones[2]:
                self.schedule_lr()
            for imgs, labels in tqdm(iter(self.loader)):
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)
                loss = conf.ce_loss(thetas, labels)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.

                if self.step % self.evaluate_every == 0 and self.step != 0:
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30,
                                                                               self.agedb_30_issame)
                    self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
                    self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame)
                    self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
                    self.model.train()
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)

                self.step += 1

        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def embedding_face(self, img, model):
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

    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)

    def euclid_distance(self, source_embedding, target_embedding):
        if isinstance(source_embedding, list):
            source_embedding = np.array(source_embedding)

        if isinstance(target_embedding, list):
            target_embedding = np.array(target_embedding)

        diff = np.subtract(source_embedding, target_embedding)
        dist = np.sum(np.square(diff), 1)
        return dist

    def infer_csv(self, conf, faces, representations, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                if conf.network in ['r100', 'vit', 'r34', 'r18', 'mbf']:
                    emb = self.embedding_face(img, self.model)
                    embs.append(emb.cpu().detach().numpy())
                else:
                    mirror = trans.functional.hflip(img)
                    emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                    emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                    embs.append(l2_norm(emb + emb_mirror))

            else:
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))


        results = []
        df = pd.DataFrame(representations, columns=["identity", f"emb", "folder_path"])
        for emb_target in embs:
            distances = []
            for index, instance in df.iterrows():
                emb_source = instance['emb']
                if conf.network in ['r100', 'vit', 'r34', 'r18', 'mbf']:
                    dist = self.euclid_distance(emb_source, emb_target)
                else:
                    dist = self.euclid_distance(emb_source, emb_target.cpu().detach().numpy())
                distances.append(dist[0])

            df['distances'] = distances
            df = df[df['distances'] < self.threshold]

            if len(df) > 0:
                df = df.sort_values(
                    by=["distances"], ascending=True
                ).reset_index(drop=True)
                df = df.drop(['emb'], axis=1)
                results.append(df.head(1).to_dict())
            else:
                return [{
                    'identity': {
                        0: 'Unknown'
                    },
                    'folder_path': {
                        0: ''
                    },
                    'distances': {
                        0: np.array(distances).min()
                    },
                }]

        return results
    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                if conf.network in ['r100', 'vit', 'r34', 'r18', 'mbf']:
                    emb = self.embedding_face(img, self.model)
                    embs.append(emb)
                else:
                    mirror = trans.functional.hflip(img)
                    emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                    emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                    embs.append(l2_norm(emb + emb_mirror))

            else:
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))

        source_embs = torch.cat(embs)
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1  # if no match, set idx to -1
        return min_idx, minimum
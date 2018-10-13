import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from gazenet import GazeNet

import time
import os
import numpy as np
import json
import cv2
from PIL import Image, ImageOps
import random
from tqdm import tqdm
import operator
import itertools
from scipy.io import  loadmat, savemat
import logging

from scipy import signal

from utils import data_transforms
from utils import get_paste_kernel, kernel_map

# log setting
log_dir = 'log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = log_dir + 'test_ourdata.log'

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s',
                    filename=log_file,
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


class GazeDataset(Dataset):
    def __init__(self, root_dir, mat_file, training='train'):
        assert (training in set(['train', 'test']))
        self.root_dir = root_dir
        self.mat_file = mat_file
        self.jsonData = json.load(open(self.mat_file))
        self.training = training

        self.bboxes = self.jsonData['boxes']
        self.gazes = self.jsonData['points']
        self.paths = self.jsonData['path']
        self.image_num = len(self.paths)

        logging.info('%s contains %d images' % (self.mat_file, self.image_num))

    def generate_data_field(self, eye_point):
        """eye_point is (x, y) and between 0 and 1"""
        height, width = 224, 224
        x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
        y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
        grid = np.stack((x_grid, y_grid)).astype(np.float32)

        x, y = eye_point
        x, y = x * width, y * height

        grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
        norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
        # avoid zero norm
        norm = np.maximum(norm, 0.1)
        grid /= norm
        return grid

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        image_path = os.path.join(self.root_dir, image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        box = self.bboxes[idx]
        point = self.gazes[idx]
        h, w = image.shape[:2]
        eye = [(box[0] + 0.5 * box[2]) / w, (box[1] + 0.5 * box[3]) /h ]
        gaze = [point[0] / w, point[1] / h]


        if random.random() > 0.5 and self.training == 'train':
            eye = [1.0 - eye[0], eye[1]]
            gaze = [1.0 - gaze[0], gaze[1]]
            image = cv2.flip(image, 1)
            
        # crop face
        x_c, y_c = eye
        x_0 = x_c - 0.15
        y_0 = y_c - 0.15
        x_1 = x_c + 0.15
        y_1 = y_c + 0.15
        if x_0 < 0:
            x_0 = 0
        if y_0 < 0:
            y_0 = 0
        if x_1 > 1:
            x_1 = 1
        if y_1 > 1:
            y_1 = 1
        h, w = image.shape[:2]
        face_image = image[int(y_0 * h):int(y_1 * h), int(x_0 * w):int(x_1 * w), :]
        # process face_image for face net
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_image)
        face_image = data_transforms[self.training](face_image)
        # process image for saliency net
        #image = image_preprocess(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = data_transforms[self.training](image)

        # generate gaze field
        gaze_field = self.generate_data_field(eye_point=eye)
        # generate heatmap
        heatmap = get_paste_kernel((224 // 4, 224 // 4), gaze, kernel_map, (224 // 4, 224 // 4))
        '''
        direction = gaze - eye
        norm = (direction[0] ** 2.0 + direction[1] ** 2.0) ** 0.5
        if norm <= 0.0:
            norm = 1.0

        direction = direction / norm
        '''
        sample = {'image' : image,
                  'face_image': face_image,
                  'eye_position': torch.FloatTensor(eye),
                  'gaze_field': torch.from_numpy(gaze_field),
                  'gt_position': torch.FloatTensor(gaze),
                  'gt_heatmap': torch.FloatTensor(heatmap).unsqueeze(0)}

        return sample


def test(net, test_data_loader):
    net.eval()
    total_loss = []
    total_error = []
    info_list = []
    heatmaps = []

    for data in test_data_loader:
        image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = \
            data['image'], data['face_image'], data['gaze_field'], data['eye_position'], data['gt_position'], data['gt_heatmap']
        image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = \
            map(lambda x: Variable(x.cuda(), volatile=True), [image, face_image, gaze_field, eye_position, gt_position, gt_heatmap])

        direction, predict_heatmap = net([image, face_image, gaze_field, eye_position])

        heatmap_loss, m_angle_loss = \
            F_loss(direction, predict_heatmap, eye_position, gt_position, gt_heatmap)

        loss = heatmap_loss + m_angle_loss


        total_loss.append([heatmap_loss.data[0],
                          m_angle_loss.data[0], loss.data[0]])
        logging.info('loss: %.5lf, %.5lf, %.5lf'%( \
              heatmap_loss.data[0], m_angle_loss.data[0], loss.data[0]))

        middle_output = direction.cpu().data.numpy()
        final_output = predict_heatmap.cpu().data.numpy()
        target = gt_position.cpu().data.numpy()
        eye_position = eye_position.cpu().data.numpy()
        for m_direction, f_point, gt_point, eye_point in \
            zip(middle_output, final_output, target, eye_position):
            f_point = f_point.reshape([224 // 4, 224 // 4])
            heatmaps.append(f_point)

            h_index, w_index = np.unravel_index(f_point.argmax(), f_point.shape)
            f_point = np.array([w_index / 56., h_index / 56.])

            f_error = f_point - gt_point
            f_dist = np.sqrt(f_error[0] ** 2 + f_error[1] ** 2)

            # angle 
            f_direction = f_point - eye_point
            gt_direction = gt_point - eye_point

            norm_m = (m_direction[0] **2 + m_direction[1] ** 2 ) ** 0.5
            norm_f = (f_direction[0] **2 + f_direction[1] ** 2 ) ** 0.5
            norm_gt = (gt_direction[0] **2 + gt_direction[1] ** 2 ) ** 0.5
            
            m_cos_sim = (m_direction[0]*gt_direction[0] + m_direction[1]*gt_direction[1]) / \
                        (norm_gt * norm_m + 1e-6)
            m_cos_sim = np.maximum(np.minimum(m_cos_sim, 1.0), -1.0)
            m_angle = np.arccos(m_cos_sim) * 180 / np.pi

            f_cos_sim = (f_direction[0]*gt_direction[0] + f_direction[1]*gt_direction[1]) / \
                        (norm_gt * norm_f + 1e-6)
            f_cos_sim = np.maximum(np.minimum(f_cos_sim, 1.0), -1.0)
            f_angle = np.arccos(f_cos_sim) * 180 / np.pi

            
            total_error.append([f_dist, m_angle, f_angle])
            info_list.append(list(f_point))

    #info_list = np.array(info_list)
    #np.savez('multi_scale_concat_prediction.npz', info_list=info_list)

    #heatmaps = np.stack(heatmaps)
    #np.savez('multi_scale_concat_heatmaps.npz', heatmaps=heatmaps)

    logging.info('average loss : %s'%str(np.mean(np.array(total_loss), axis=0)))
    logging.info('average error: %s'%str(np.mean(np.array(total_error), axis=0)))
    error_list = np.array(total_error)
    print(error_list.shape)
    #np.savez('error_our_data_our_method.npz', error_list=error_list)
    #savemat("error_our_data_our_method.mat", {"distance":error_list[:, 0], "angle":error_list[:, 2]})

    net.train()
    return


def main():

    test_set = GazeDataset(root_dir='../OurData/',
                           mat_file='../OurData/annotation.json',
                           training='test')
    test_data_loader = DataLoader(test_set, batch_size=1,
                                  shuffle=False, num_workers=8)

    net = GazeNet()
    net = DataParallel(net)
    net.cuda()

    pretrained_dict = torch.load('../model/pretrained_model.pkl')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    test(net, test_data_loader)


if __name__ == '__main__':
    main()

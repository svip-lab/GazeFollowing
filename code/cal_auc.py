from scipy.io import loadmat
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

test_mat_file = '../GazeFollowData/test2_annotations.mat'
prediction_file = 'multi_scale_concat_heatmaps.npz'

anns = loadmat(test_mat_file)
gazes = anns['test_gaze']
eyes = anns['test_eyes']
N = anns['test_path'].shape[0]

prediction = np.load(prediction_file)['heatmaps']
print(prediction.shape)

gt_list, pred_list = [], []
error_list = []
for i in range(N):
    pred = prediction[i, :, :]
    eye_point = eyes[0, i][0]
    gt_points = gazes[0, i]
    pred = cv2.resize(pred, (5, 5))
    #pred[...] = 0.0
    #pred[2, 2] = 1.0
    gt_heatmap = np.zeros((5, 5))
    for gt_point in gt_points:
        x, y = list(map(int, list(gt_point * 5)))
        gt_heatmap[y, x] = 1.0

    score = roc_auc_score(gt_heatmap.reshape([-1]).astype(np.int32), pred.reshape([-1]))
    error_list.append(score)
    gt_list.append(gt_heatmap)
    pred_list.append(pred)

print("mean", np.mean(error_list))
gt_list = np.stack(gt_list).reshape([-1])
pred_list = np.stack(pred_list).reshape([-1])

print("auc score")
score = roc_auc_score(gt_list, pred_list)
print(score)

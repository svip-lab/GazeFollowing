# Gaze following
PyTorch implementation of our ACCV2018 paper:

**'Believe It or Not, We Know What You Are Looking at!'** [[paper](https://arxiv.org/pdf/1907.02364.pdf)]
[[poster](images/poster.pdf)]

Dongze Lian*, Zehao Yu*, Shenghua Gao
 
(* Equal Contribution)

# Prepare training data

GazeFollow dataset is proposed in [1], please download the dataset from http://gazefollow.csail.mit.edu/download.html.
Note that the downloaded testing data may have wrong label, so we request test2 provided by author. 
I do not know whether the author update their testing set. If not, it is better for you to e-mail authors in [1]. 
For your convenience, we also paste the testing set link [here](http://videogazefollow.csail.mit.edu/downloads/test_set.zip) provided by authors in [1] when we request. 
(Note that the license is in [1])


# Download our dataset
OurData is in [Onedrive](https://yien01-my.sharepoint.com/:u:/g/personal/doubility_z0_tn/Ea2BrlvFfQ5Dt8UjgfVnA6QB7yUAvbDDQFr1rZ_b0m9Nvw?e=jaUGWb)
Please download and unzip it

OurData contains data descriped in our paper.
```
OurData/tools/extract_frame.py
``` 
extract frame from clipVideo in 2fps.
Different version of ffmpeg may have different results, we provide our extracted images.
```
OurData/tools/create_video_image_list.py
``` 
extract annotation to json.

# Testing on gazefollow data
```
cd code
python test_gazefollow.py
```

# Test on our data
Please download the [pretrained model](https://drive.google.com/open?id=1eN0NysvRNsWaoyJea3w1Tdbt7iPMvjmp) manually and save to model/
```
cd code
python test_ourdata.py
```

# Training scratch
```
cd code
python train.py
```


# Reference:
[1] Recasens*, A., Khosla*, A., Vondrick, C., Torralba, A.: Where are they looking? In: Advances in Neural
Information Processing Systems (NIPS) (2015).



# Citation
If this project is helpful for you, you can cite our paper:
```
@InProceedings{Lian_2018_ACCV,
author = {Lian, Dongze and Yu, Zehao and Gao, Shenghua},
title = {Believe It or Not, We Know What You Are Looking at!},
booktitle = {ACCV},
year = {2018}
}
```
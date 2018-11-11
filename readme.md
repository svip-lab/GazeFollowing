# prepare training data
GazeFollowData contains training and testing data from [1], please download data form http://gazefollow.csail.mit.edu/download.html.
Note that the downloaded testing data may have wrong label, so we provided test2 provided from author.

# download our dataset
OurData contains data descriped in our paper.
OurData/tools/extract_frame.py extract frame from clipVideo in 2fps.
Different version of ffmpeg may have different results, we provide our extracted images.
OurData/tools/create_video_image_list.py extract annotation to json.

# testing on gazefollow data
cd code
python test_gazefollow.py


# test on our data
please download the pretrained model from https://drive.google.com/open?id=1eN0NysvRNsWaoyJea3w1Tdbt7iPMvjmp manually as save to model/

cd code
python test_ourdata.py

# training scratch
cd code
python train.py

#TODO:



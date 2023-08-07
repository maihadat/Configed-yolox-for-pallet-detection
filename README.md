## A pytorch easy re-implement of "YOLOX: Exceeding YOLO Series in 2021"


## 1. Environment

    pytorch>=1.7.0, python>=3.6, Ubuntu/Windows, see more in 'requirements.txt'
    
    To download the requirements, just run the YOLOX.ipynb, the Requirements section

## 2. Notes
    
    for yolox implementation in rendering pallet detections, we have implements train and predict parts in YOLOX.ipynb, 
    all the need is to download and unzip the Data.rar from the link in the data section, and then put it in the 
    Configed-yolox-for-pallet-detection folder(because the data is too large so we cann't put it in github repo).
    You can see the details in the below sections.

## 3. Object Detection

#### Model Zoo

All pretrained weights with COCO dataset can be downloaded
from [GoogleDrive](https://drive.google.com/drive/folders/1qEMLzikH5JwRNRoHpeCa6BJBeSQ6xXCH?usp=sharing)
or [BaiduDrive](https://pan.baidu.com/s/1UsbdnyVwRJhr9Vy1tmJLeQ) (code:bc72)

|Model      |test size  |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 | Params<br>(M) |
| ------    |:---:      |:---:                   | :---:                   |:---:          |
|yolox-nano |416        |25.4                    |25.7                     |0.91           |
|yolox-tiny |416        |33.1                    |33.2                     |5.06           |
|yolox-s    |640        |39.3                    |39.6                     |9.0            |
|yolox-m    |640        |46.2                    |46.4                     |25.3           |
|yolox-l    |640        |49.5                    |50.0                     |54.2           |
|yolox-x    |640        |50.5                    |51.1                     |99.1           |
|yolox-x    |800        |51.2                    |51.9                     |99.1           |

mAP was reevaluated on COCO val2017 and test2017, and some results are slightly better than the official
implement [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX). You can reproduce them by scripts in 'evaluate.sh'

#### Dataset

    download Data.rar from Google Drive, this folder contains render pallet data in YOLO format, link below:
    https://drive.google.com/file/d/1pKL2N7nAqEXMM1vB4PHgKVpkHAWCmHnl/view?usp=sharing
    
    unzip and put dataset in Configed-yolox-for-pallet-detection folders.

#### Train

    See more example in 'train.sh'
    a. Train from scratch:(backbone="CSPDarknet-s" means using yolox-s, and you can change it, eg: CSPDarknet-nano, tiny, s, m, l, x)
    python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=True val_intervals=2 data_num_workers=6 batch_size=48
    
    b. Finetune, download pre-trained weight on COCO and finetune on customer dataset:
    python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=True val_intervals=2 data_num_workers=6 batch_size=48 load_model="../weights/yolox-s.pth"
    
    c. Resume, you can use 'resume=True' when your training is accidentally stopped:
    python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=True val_intervals=2 data_num_workers=6 batch_size=48 load_model="exp/coco_CSPDarknet-s_640x640/model_last.pth" resume=True

    In the YOLOX.ipynb file we have use finetune method to take advantage of pretrained weights from yolox-s model.


#### Evaluate

    Module weights will be saved in './exp/your_exp_id/model_xx.pth'
    change 'load_model'='weight/path/to/evaluate.pth' and backbone='backbone-type' in 'evaluate.sh'
    sh evaluate.sh

    We also implement some chart drawing and iou, confidents calculating in YOLOX.ipynb

#### Predict/Inference/Demo

    a. Predict images, change img_dir and load_model
    python predict.py gpus='0' backbone="CSPDarknet-s" vis_thresh=0.3 load_model="exp/coco_CSPDarknet-s_640x640/model_best.pth" img_dir='/path/to/dataset/images/val2017'
    
    b. Predict video
    python predict.py gpus='0' backbone="CSPDarknet-s" vis_thresh=0.3 load_model="exp/coco_CSPDarknet-s_640x640/model_best.pth" video_dir='/path/to/your/video.mp4'
    
    You can also change params in 'predict.sh', and use 'sh predict.sh'

    In the YOLOX.ipynb file we have implements predicting sections on the test data of the rendering pallets dataset.

## 4. Acknowledgement

    https://github.com/Megvii-BaseDetection/YOLOX
    https://github.com/PaddlePaddle/PaddleDetection
    https://github.com/open-mmlab/mmdetection
    https://github.com/xingyizhou/CenterNet

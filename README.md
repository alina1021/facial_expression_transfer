# facial_expression_transfer

This is a pix2pix demo that learns from facial landmarks and translates this into a face. A webcam-enabled application is also provided that translates your face to the trained face in real-time.

This project comes in three repositories. This repository, for general purpose scripts and documentation, and a forked version of the [face2face](https://github.com/datitran/face2face-demo) and [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow)  repositories. 
# The presentation slides for this project are provided as [Google Slides (https://docs.google.com/presentation/d/10UUUsJWL2R0r9y5FpoIfv9wJboyUFt7svuSvhGRsmZ8/edit?usp=sharing).

## Getting Started

### 1. Prepare Environment

```
# Clone this repo
git clone https://github.com/alina1021/facial_expression_transfer.git --recursive

# Create the conda environment from file (Mac OSX)
conda env create -f environment.yml
```

### 2. Generate Training Data

```
python generate_train_data.py --file angela_merkel_speech.mp4 --num 400 --landmark-model shape_predictor_68_face_landmarks.dat
```

Input:

- `file` is the name of the video file from which you want to create the data set.
- `num` is the number of train data to be created.
- `landmark-model` is the facial landmark model that is used to detect the landmarks. A pre-trained facial landmark model is provided [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).

Output:

- Two folders `original` and `landmarks` will be created.

If you want to download my dataset, here is also the [video file](https://u7410512.dl.dropboxusercontent.com/u/7410512/face2face-demo/angela_merkel_speech.mp4) that I used and the generated [training dataset](https://u7410512.dl.dropboxusercontent.com/u/7410512/face2face-demo/dataset.zip) (400 images already split into training and validation).

### 3. Train Model

```
# Clone the repo from Christopher Hesse's pix2pix TensorFlow implementation
git clone https://github.com/affinelayer/pix2pix-tensorflow.git

# Move the original and landmarks folder into the pix2pix-tensorflow folder
mv face2face-demo/landmarks face2face-demo/original pix2pix-tensorflow/photos

# Go into the pix2pix-tensorflow folder
cd pix2pix-tensorflow/

# Resize original images
python tools/process.py \
  --input_dir photos/original \
  --operation resize \
  --size 1024 \
  --output_dir photos/original_resized

# Resize landmark images
python tools/process.py \
  --input_dir photos/landmarks \
  --operation resize \
  --size 1024 \
  --output_dir photos/landmarks_resized

# Combine both resized original and landmark images
python tools/process.py \
  --input_dir photos/landmarks_resized \
  --b_dir photos/original_resized \
  --operation combine \
  --output_dir photos/combined

# Split into train/val set
python tools/split.py \
  --dir photos/combined

# Train the model on the data
python pix2pix.py \
  --mode train \
  --output_dir face2face-model \
  --max_epochs 200 \
  --input_dir photos/combined/train \
  --which_direction AtoB
```

For more information around training, have a look at Christopher Hesse's [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow) implementation.

### 4. Export Model

1. First, we need to reduce the trained model so that we can use an image tensor as input:
    ```
    python ../reduce_model.py --model-input face2face-model --model-output face2face-reduced-model
    ```

    Input:

    - `model-input` is the model folder to be imported.
    - `model-output` is the model (reduced) folder to be exported.

    Output:

    - It returns a reduced model with less weights file size than the original model.

2. Second, we freeze the reduced model to a single file.
    ```
    python ../freeze_model.py --model-folder face2face-reduced-model
    ```

    Input:

    - `model-folder` is the model folder of the reduced model.

    Output:

    - It returns a frozen model file `frozen_model.pb` in the model folder.

### 5. Run Demo

```
python run_webcam.py --source 0 --show 0 --landmark-model shape_predictor_68_face_landmarks.dat --tf-model face2face-reduced-model/frozen_model.pb
```

Input:

- `source` is the device index of the camera (default=0).
- `show` is an option to either display the normal input (0) or the facial landmark (1) alongside the generated image (default=0).
- `landmark-model` is the facial landmark model that is used to detect the landmarks.
- `tf-model` is the frozen model file.



## Models

### Dat Tran - Angela Merkel 256x256

![example](example.gif)

**THIS MODEL WILL NOT WORK WITH THIS CODE**

Pre-trained frozen model [here](https://dl.dropboxusercontent.com/s/rzfaoeb3e2ta343/face2face_model_epoch_200.zip). This model is trained on 400 images with epoch 200.

### Karol Majek - Krzysztof Gonciarz 1024x1024

Me, my face and output:

![Face2Face Gonciarz](example-gonciarz.gif)


Video on youtube:

[![Face2Face Krzysztof Gonciarz YAPZTAJ KASIĘ CICHOPEK 4 - Zapytaj Beczkę #146](http://img.youtube.com/vi/v5VDJKCrP6A/0.jpg)](http://www.youtube.com/watch?v=v5VDJKCrP6A)

Frozen model can be downloaded from [here](https://goo.gl/8BgnXA) (1.1GB)

## Requirements
- [Anaconda / Python 3.5](https://www.continuum.io/downloads)
- [TensorFlow 1.2](https://www.tensorflow.org/)
- [OpenCV 3.0](http://opencv.org/)
- [Dlib 19.4](http://dlib.net/)

## Acknowledgments

Thanks to [Dat Tran](http://www.dat-tran.com/) for inspiration, code and model!

## Copyright

See [LICENSE](LICENSE) for details.
Copyright (c) 2017 [Dat Tran](http://www.dat-tran.com/).

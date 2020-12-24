# CSAW-HackML-2020

```bash
├── data 
    └── clean_validation_data.h5 // this is clean data used to evaluate the BadNet and design the backdoor defense
    └── clean_test_data.h5
    └── sunglasses_poisoned_data.h5
├── models
    └── anonymous_bd_net.h5
    └── anonymous_bd_weights.h5
    └── sunglasses_bd_net.h5
    └── sunglasses_bd_weights.h5
├── architecture.py
└── eval.py // this is the evaluation script
```

## I. Dependencies
<<<<<<< Updated upstream
   1. Python 3.6.9 -> 3.6.12
   2. Keras 2.3.1
   3. Numpy 1.16.3
   4. Matplotlib 2.2.2
   5. H5py 2.9.0
   6. TensorFlow-gpu 1.15.2 -> 1.15.3
   
## II. Validation Data
=======

We keep most packages' major version as same as before.

```sh
pip3 install -r requirements.txt
```

## II. How To Run

Please replace `model_name` with one of the following options: `sunglasses`, `anonymous_1`, `anonymous_2`, `multi_trigger_multi_target`.

### 1. Visualize & Reverse Engineer the Trigger

*estimated visualization time: 170 mins on Tesla T4 for each model.*

You can skip the visualization part by downloading the results from [here](https://drive.google.com/drive/folders/18vAKWeiGGFdf2mw6EX1rFduSAHk0XL9i?usp=sharing).

```shell
python3 visualize_example.py $model_name
```

### 2. Detect Targeted Label

```shell
python3 mad_outlier_detection.py $model_name
```

### 3. Repair Backdoored Model

*estimated repair time: 20 mins on Tesla T4 for each model.*

You can also mannully redo the prune and repair by deleting the model under `models` folder with `pruned` or `repair` in the model name.

```shell
python3 repair_model.py $model_name
```

### 3. Repair Backdoored Model

*estimated repair time: 20 mins on Tesla T4 for each model.*

You can also mannully redo the prune and repair by deleting the model under `models` folder with `pruned` or `repair` in the model name.

```shell
python3 repair_model.py $model_name
```

### 4. STRIP Method

In `strip.py`, we implement method in `STRIP: A Defence Against Trojan Attacks on DeepNeural Networks` to detect trojaned input.

`detect_trojan`, `detect_trojan_batch` in `strip.py` can detect whether the input is trojaned or not for single input and a batch of inputs respectively, which will return label N, if the input is trojaned and return label 0~(N-1), if the input is clean.

The `eval_strip_[badnet_name].py` is script to evaluate.

```shell
python3 eval_strip_sunglasses.py $image_path
python3 eval_strip_anonymous_1.py $image_path
python3 eval_strip_anonymous_2.py $image_path
python3 eval_strip_multi.py $image_path
```

## III. Validation Data

>>>>>>> Stashed changes
   1. Download the validation and test datasets from [here](https://drive.google.com/drive/folders/13o2ybRJ1BkGUvfmQEeZqDo1kskyFywab?usp=sharing) and store them under `data/` directory.
   2. The dataset contains images from YouTube Aligned Face Dataset. We retrieve 1283 individuals each containing 9 images in the validation dataset.
   3. sunglasses_poisoned_data.h5 contains test images with sunglasses trigger that activates the backdoor for sunglasses_bd_net.h5.

## III. Evaluating the Backdoored Model
   1. The DNN architecture used to train the face recognition model is the state-of-the-art DeepID network. This DNN is backdoored with multiple triggers. Each trigger is associated with its own target label. 
   2. To evaluate the backdoored model, execute `eval.py` by running:  
      `python3 eval.py <clean validation data directory> <model directory>`.
      
      E.g., `python3 eval.py data/clean_validation_data.h5  models/sunglasses_bd_net.h5`.
   3. Clean data classification accuracy on the provided validation dataset for sunglasses_bd_net.h5 is 97.87 %.

## IV. Evaluating the Submissions
To aid teams in designing their defense, here are a few guidelines to keep in mind to get maximum points for the submission:  
   1. Defense should generalize well to other backdoored networks. To verify the defense generalizability, the organizers will evaluate the submission on a specially curated BadNet, anonymous_bd_net.h5, with different trigger properties. 
   2. Teams gain maximum points if the defense greatly reduces attack success rate on the trigger(s) while maintaining high clean classification accuracy.
   3. Points will also be given to teams that identify poisoned images in the online test stream of images.
   4. Fewer points will be allocated to teams that only detect the network as clean or backdoored.
   5. Report should contain a description of the defense performance on adaptive attackers.


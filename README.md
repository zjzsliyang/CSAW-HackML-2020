# CSAW-HackML-2020

```bash
├── data 
    └── clean_validation_data.h5 // this is clean data used to evaluate the BadNet and design the backdoor defense
    └── clean_test_data.h5
    └── sunglasses_poisoned_data.h5
    └── anonymous_1_poisoned_data.h5
    └── Multi-trigger Multi-target
        └── eyebrows_poisoned_data.h5
        └── lipstick_poisoned_data.h5
        └── sunglasses_poisoned_data.h5
├── models
    └── sunglasses_bd_net.h5
    └── sunglasses_bd_weights.h5
    └── multi_trigger_multi_target_bd_net.h5
    └── multi_trigger_multi_target_bd_weights.h5
    └── anonymous_1_bd_net.h5
    └── anonymous_1_bd_weights.h5
    └── anonymous_2_bd_net.h5
    └── anonymous_2_bd_weights.h5
├── architecture.py
├── eval.py // this is the evaluation script
├   // add following
├── documents
├   // Neural Cleanse Approach
├── requirements.txt // package version used
├── config.yaml // configuration
├── results // will generated after visualization
├── utils.py
├── visualizer.py
├── visualize_example.py
├── mad_outlier_detection.py
├── repair_model.py
├   // Neural Cleanse Approach
├── eval_strip_sunglasses.py
├── eval_strip_anonymous_1.py
├── eval_strip_anonymous_2.py
├── eval_strip_multi.py
└── strip.py
```

## I. Dependencies
We keep most packages' major version as same as before.

```bash
pip3 install -r requirements.txt
```

## II. How To Run

We implemented two approachs to detect and repair the backdoored model, i.e. [Neural Cleanse](https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf) and [STRIP](https://arxiv.org/pdf/1902.06531.pdf).

### Neural Cleanse

Please replace `model_name` with one of the following options: `sunglasses`, `anonymous_1`, `anonymous_2`, `multi_trigger_multi_target`.

#### 1. Visualize & Reverse Engineer the Trigger

*estimated visualization time: 170 mins on Tesla T4 for each model.*

You can skip the visualization part by downloading the results from [here](https://drive.google.com/drive/folders/18vAKWeiGGFdf2mw6EX1rFduSAHk0XL9i?usp=sharing).

```shell
python3 visualize_example.py $model_name
```

#### 2. Detect Targeted Label

```shell
python3 mad_outlier_detection.py $model_name
```

#### 3. Repair Backdoored Model

*estimated repair time: 20 mins on Tesla T4 for each model.*

You can also mannully redo the prune and repair by deleting the model under `models` folder with `pruned` or `repair` in the model name.

```shell
python3 repair_model.py $model_name
```

### STRIP

`detect_trojan`, `detect_trojan_batch` in `strip.py` can detect whether the input is trojaned or not for single input and a batch of inputs respectively, which will return label $N$, if the input is trojaned and return label in $ [0 , (N-1)]$, if the input is clean.

The `eval_strip_[badnet_name].py` is script to evaluate.

```shell
python3 eval_strip_sunglasses.py $image_path
python3 eval_strip_anonymous_1.py $image_path
python3 eval_strip_anonymous_2.py $image_path
python3 eval_strip_multi.py $image_path
```

## III. Validation Data

   1. Download the validation and test datasets from [here](https://drive.google.com/drive/folders/13o2ybRJ1BkGUvfmQEeZqDo1kskyFywab?usp=sharing) and store them under `data/` directory.
   2. The dataset contains images from YouTube Aligned Face Dataset. We retrieve 1283 individuals each containing 9 images in the validation dataset.
   3. sunglasses_poisoned_data.h5 contains test images with sunglasses trigger that activates the backdoor for sunglasses_bd_net.h5. Similarly, there are other .h5 files with poisoned data that correspond to different BadNets under models directory.

## IV. Evaluating the Backdoored Model
   1. The DNN architecture used to train the face recognition model is the state-of-the-art DeepID network. This DNN is backdoored with multiple triggers. Each trigger is associated with its own target label. 
   2. To evaluate the backdoored model, execute `eval.py` by running:  
      `python3 eval.py <clean validation data directory> <model directory>`.

      E.g., `python3 eval.py data/clean_validation_data.h5  models/sunglasses_bd_net.h5`. Clean data classification accuracy on the provided validation dataset for sunglasses_bd_net.h5 is 97.87 %.

## V. Evaluating the Submissions
The teams should submit a single eval.py script for each of the four BadNets provided to you. In other words, your submission should include four eval.py scripts, each corresponding to one of the four BadNets provided. YouTube face dataset has classes in range [0, 1282]. So, your eval.py script should output a class in range [0, 1283] for a test image w.r.t. a specific backdoored model. Here, output label 1283 corresponds to poisoned test image and output label in [0, 1282] corresponds to the model's prediction if the test image is not flagged as poisoned. Effectively, design your eval.py with input: a test image (in png or jpeg format), output: a class in range [0, 1283]. Output 1283 if the test image is poisoned, else, output the class in range [0,1282].

Teams should submit their solutions using GitHub. All your models (and datasets) should be uploaded to the GitHub repository. If your method relies on any dataset with large size, then upload the data to a shareable drive and provide the link to the drive in the GitHub repository. To efficiently evaluate your work, provide a README file with clear instructions on how to run the eval.py script with an example.
For example: `python3 eval_anonymous_2.py data/test_image.png`. Here, eval_anonymous_2.py is designed for anonynous_2_bd_net.h5 model. Output should be either 1283 (if test_image.png is poisoned) or one class in range [0, 1282] (if test_image.png is not poisoned).
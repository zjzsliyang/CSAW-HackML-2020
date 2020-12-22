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
├── requirements.txt // package version used
├── config.yaml // configuration
├── results // will generated after visualization
├── utils.py
├── visualizer.py
├── visualize_example.py
├── mad_outlier_detection.py
└── prune_model.py
```

## I. Dependencies

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

### 3. Prune Backdoored Model & Evaluate the Result

*estimated prune time: 10 mins on Tesla T4 for each model.*

You can also skip the prune part by downloading the pruned models from [here](https://drive.google.com/drive/folders/1OXLFlGVqfFy6Lq22Qz5pLmWmRViv-OLD?usp=sharing) (with `pruned` in the model name) and still run the following code to see the result. 

Note that currently we can not evaluate `anonymous_2` and `multi_trigger_multi_target` since dataset missing or it make no sense in `multi_trigger_multi_target` part.

```shell
python3 prune_model.py $model_name
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

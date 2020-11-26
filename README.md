# AutoGenOfContrastSetsFromSceneGraphs
Repository for the paper "Automatic Generation of Contrast Sets from Scene Graphs"

![](https://github.com/yonatanbitton/AutoGenOfContrastSetsFromSceneGraphs/blob/main/fig1.png?raw=true | width=100)


## Setup   

First, download the data directory.    
`https://drive.google.com/drive/folders/1BjPShPRcU7B-rh8I6WOBm20zjBv31E2R?usp=sharing`  
  
Required packages are `numpy, pandas, cv2`   

Config file - `config.py`.    
please change the current data path (`data_dir = "/Users/yonatab/data/gqa_data"`) to your data directory.  
In addition, change the max number of augmentations desired in the line of `number_of_augmentations_to_produce = 1`  

The GQA data is taken from here: https://cs.stanford.edu/people/dorarad/gqa/download.html  
The questions being used are the balanced questions. The questions and scene graphs available in the drive, please download the images from the link above.  

## Structure     

Code for the ***question templates*** is available at `src/question_templates`.  

Code for the ***valid augmentation*** is available at `src/work_on_valid/augment_gqa`.  
 
Code for the ***train augmentation*** is available at `src/work_on_train/augment_gqa`.  
  
***Notice***: to create the final files for lxmert prediction there is a process of removing the duplicate files (that are both in the GQA and the augmentation).  

Change the `question_template` flag to choose the question template you want to work on, for example:  
```python src/work_on_valid/augment_gqa.py --question_template do_you_see_x_or_y```  

Expected output:
```
AUGMENT_TRAIN: False
first_exp_1_outputs_for_match
template_class_constructor: do_you_see_x_or_y
Loaded 132062 val examples
Got val data
idx: 23, image_id: 2378230, qa_key: 0953520
Do you see skateboards or boys?, no
Do you see jeans or boys?, yes

idx: 25, image_id: 2378235, qa_key: 0959247
Do you see either any ostrich or penguin there?, no
Do you see either any ostrich or cow there?, yes
...
...
```

In addition, change the `AUGMENT_TRAIN` variable to `True` if you want to augment the train. Else - it is `False`.  

Expected output:
```
AUGMENT_TRAIN: True
first_exp_1_outputs_for_match
template_class_constructor: do_you_see_x_or_y
RUNNING AUGMENTATION ON TRAIN
Loaded 943000 train examples
Got train data
idx: 27, image_id: 2410695, qa_key: 08371629
Do you see either a yellow giraffe or bird?, yes
Do you see either a yellow giraffe or tree?, no

idx: 27, image_id: 2410695, qa_key: 08371958
Do you see any yellow giraffes or birds?, yes
Do you see any yellow giraffes or wings?, no
...
...
```

The final files (after removing duplicates) are the directories called `final_files_lxmert_format` (In each corresponding directory - train/validation, max. augs: 1/3/5).    
The script that performs this post-process is `create_testdev_of_augmented_dataset.py`.  

## Model predictions

Use the public implementations of LXMERT and MAC     
https://github.com/airsplay/lxmert   
https://github.com/stanfordnlp/mac-network/tree/gqa   
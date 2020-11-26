# AutoGenOfContrastSetsFromSceneGraphs
Repository for the paper "Automatic Generation of Contrast Sets from Scene Graphs"

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
  
***Notice***: to create the final files for lxmert prediction there is a process of removing the duplicate files (that are both in the GQA and the augmentation)  

Change the `question_template` flag to choose the question template you want to work on, for example:  
```python src/work_on_valid/augment_gqa.py --question_template do_you_see_x_or_y```  

The final files (after removing duplicates) are the directories called `final_files_lxmert_format` (In each corresponding directory - train/validation, max. augs: 1/3/5).    
The script that performs this post-process is `create_testdev_of_augmented_dataset`.  

## Model predictions

Use the public implementations of LXMERT and MAC     
https://github.com/airsplay/lxmert   
https://github.com/stanfordnlp/mac-network/tree/gqa   
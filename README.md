# Automatic Generation of Contrast Sets from Scene Graphs
Repository for the paper "Automatic Generation of Contrast Sets from Scene Graphs", accepted to NAACL 2021.
https://arxiv.org/abs/2103.09591

## Intro   

Our method leverages rich semantic input representation to automatically generate contrast sets for the visual question answering task. Our method computes the answer of perturbed questions, thus vastly reducing annotation cost and enabling thorough evaluation of models’ performance on various semantic aspects (e.g., spatial or relational reasoning).  

This repository allows perturbation of the GQA validation set questions, by changing a single atom (object, attribute, or relationship) to a different atom, causing the answer to be changed as well. For example - changing "bird" to "tree", and the answer from "yes" to "no" (More examples below):
```
Is there a fence near the puddle?, yes
Is there an elephant near the puddle?, no
```

<img src="https://i.ibb.co/JsY90hj/fig1.png" width="350">

### Contrast sets in LXMERT input format
If you only want the contrast sets in LXMERT input format, an example for 1 augmentation per question is available here: https://drive.google.com/drive/folders/1DJuFwUnSQXuMOkYZ6e7N5_Q-V1v0nvZ4
There are 15 files. 12 files for each question template. Two more files for aggregating all of the question templates (`augmentation_all_original` and `augmentation_all_augmented`). Finally, a statistics file (`stats_first_exp_1_outputs_for_match`). 

To replicate the contrast sets construction, you will need the full setup.

## Setup   

First, download the data directory https://drive.google.com/drive/folders/1BjPShPRcU7B-rh8I6WOBm20zjBv31E2R?usp=sharing. 
  
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

**Expected output**:
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

**Expected output**:
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

## Reference

```bibtex
@inproceedings{bitton-etal-2021-automatic,
    title = "Automatic Generation of Contrast Sets from Scene Graphs: Probing the Compositional Consistency of {GQA}",
    author = "Bitton, Yonatan  and
      Stanovsky, Gabriel  and
      Schwartz, Roy  and
      Elhadad, Michael",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.9",
    pages = "94--105",
    abstract = "Recent works have shown that supervised models often exploit data artifacts to achieve good test scores while their performance severely degrades on samples outside their training distribution. Contrast sets (Gardneret al., 2020) quantify this phenomenon by perturbing test samples in a minimal way such that the output label is modified. While most contrast sets were created manually, requiring intensive annotation effort, we present a novel method which leverages rich semantic input representation to automatically generate contrast sets for the visual question answering task. Our method computes the answer of perturbed questions, thus vastly reducing annotation cost and enabling thorough evaluation of models{'} performance on various semantic aspects (e.g., spatial or relational reasoning). We demonstrate the effectiveness of our approach on the GQA dataset and its semantic scene graph image representation. We find that, despite GQA{'}s compositionality and carefully balanced label distribution, two high-performing models drop 13-17{\%} in accuracy compared to the original test set. Finally, we show that our automatic perturbation can be applied to the training set to mitigate the degradation in performance, opening the door to more robust models.",
}
```

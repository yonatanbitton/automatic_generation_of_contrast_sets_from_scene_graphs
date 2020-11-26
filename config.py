import os
import os.path as osp

data_dir = '/Users/yonatab/data/gqa_data'  # Change to your directory

number_of_augmentations_to_produce = 1 #
if number_of_augmentations_to_produce == 1:
    experiment_output_name = f'first_exp_{number_of_augmentations_to_produce}_outputs_for_match'
elif number_of_augmentations_to_produce == 3:
    experiment_output_name = f'second_exp_{number_of_augmentations_to_produce}_outputs_for_match'
elif number_of_augmentations_to_produce == 5:
    experiment_output_name = f'third_exp_{number_of_augmentations_to_produce}_outputs_for_match'

AUGMENT_TRAIN = True
print(f'AUGMENT_TRAIN: {AUGMENT_TRAIN}')
number_of_augmentations_to_produce = 1


print(experiment_output_name)

if AUGMENT_TRAIN:
    experiment_output_name += "_train"

questions_dir = osp.join(data_dir, 'questions1.2')
scene_graphs_dir = osp.join(data_dir, 'sceneGraphs')
images_dir = osp.join(data_dir, 'images')

if AUGMENT_TRAIN:
    all_experiments_output_dir = osp.join(data_dir, 'all_experiments_train')
    experiment_output_dir = osp.join(all_experiments_output_dir, f'{experiment_output_name}')
    output_dir = osp.join(experiment_output_dir, 'question_templates_outputs_train')
    output_csv_path = osp.join(output_dir, 'output_files_train')
    duplicates_dir = osp.join(experiment_output_dir, 'duplicates_train')
    preds_dir = osp.join(data_dir, 'all_experiments/gqa_lxr955_on_train_only_results_train')
else:
    all_experiments_output_dir = osp.join(data_dir, 'all_experiments')
    experiment_output_dir = osp.join(all_experiments_output_dir, f'{experiment_output_name}')
    output_dir = osp.join(experiment_output_dir, 'question_templates_outputs')
    output_csv_path = osp.join(output_dir, 'output_files')
    duplicates_dir = osp.join(experiment_output_dir, 'duplicates')
    preds_dir = osp.join(data_dir , 'all_experiments/gqa_lxr955_on_train_only_results')

for d in [all_experiments_output_dir, experiment_output_dir, output_dir, output_csv_path, duplicates_dir]:
    if not osp.exists(d):
        os.mkdir(d)


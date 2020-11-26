import difflib
import json
import os.path as osp
import pickle

import cv2
import matplotlib.pyplot as plt

from config import images_dir, scene_graphs_dir, data_dir

colors_dict = {'black': (0, 0, 0), "red": (255, 0, 0), "lime": (0, 255, 0), "blue": (0, 0, 255),
               "yellow": (255, 255, 0),
               "cyan": (0, 255, 255), "magneta": (255, 0, 255), "purple": (128, 0, 128), "orange": (255, 165, 0),
               "mediumslateblue": (106, 90, 205), "tomato": (255, 99, 71),
               "salmon": (250, 128, 114)}

colors_dict_to_hex = {'black': '#000000', "red": "FF0000", "lime": "#00FF00", "blue": "#0000FF",
               "yellow": "#FFFF00",
               "cyan": "#00FFFF", "magneta": "#FF00FF", "purple": '#800080', "orange": '#FFA500',
               "mediumslateblue": "#6A5ACD", "tomato": "#FF6347",
               "salmon": "#FA8072"}

colors_names = list(colors_dict.keys()) + ['green', 'white', 'silver', 'gold', 'gray']
colorless_objects = ['water', 'racket', 'number', 'letters', 'letter']

vowels = ['a', 'e', 'i', 'o', 'u']

SIMILARITY_THRESHOLD = 0.9
REPLACEMENT_SIMILARITY_THRESHOLD = 0.78

with open(osp.join(data_dir, 'resources', 'animals.txt'), 'r') as f:
    all_animals = [x.rstrip("\n").lower() for x in f.readlines()]
all_vehicles = ['truck', 'car', 'bus', 'train', 'tractor', 'motorcycle', 'airplane', 'helicopter', 'ship', 'boat']
all_utensil = ['knife', 'plate', 'spoon', 'fork']

scene_graphs_path = osp.join(scene_graphs_dir, 'val_sceneGraphs.json')
with open(scene_graphs_path, 'rb') as f:
    val_scene_graphs = json.load(f)
val_scene_graph_items = list(val_scene_graphs.items())

with open(osp.join(data_dir, 'val_scene_graph_objects_set.pickle'), 'rb') as f:
    val_scene_graph_objects_set = pickle.load((f))

with open(osp.join(data_dir, 'dist_for_each_scene_graph_object.pickle'), 'rb') as f:
    distribution_for_each_scene_graph_object = pickle.load(f)
existing_items_in_val = set(distribution_for_each_scene_graph_object.keys())
body_parts = ['leg', 'hand', 'wrist', 'head', 'neck', 'arm', 'finger', 'beard', 'neck', 'eye', 'ear', 'face', 'foot',
              'tail', 'thumb', 'hair', 'nose']
people_objects = ['people', 'person', 'shoe', 'pants', 'shirt', 'player']
car_objects = ['wheel', 'tire', 'license plate']
view_objects = ['sky', 'ground', 'cloud', 'clouds', 'wall', 'ceiling', 'leaves', 'leaf',
                'branch', 'branches', 'bush', 'yard', 'snow', 'sand', 'grass', 'air', 'water', 'floor']
house_objects = ['sign', 'meal', 'food', 'dish']
street_objects = ['road', 'sidewalk', 'side walk', 'number', 'numbers', 'window']

# Objects that are not usually explicitly mentioned in the scene graph, but may exists in the image.
not_replaceable_objects = body_parts + people_objects + car_objects + view_objects + house_objects + street_objects

# question to scene graph
synonyms = [('plane', 'airplane'), ('person', 'man'), ('motorbikes', 'motorcycles'), ('freezer', 'refrigerator'),
            ('garbage bin', 'trash can'), ('tub', 'bathtub'), ('shrubs', 'bushes'), ('garbage can', 'trash can'),
            ('wire', 'cord'), ('motorbike', 'motorcycle'), ('trash bin', 'trash can'), ('shrub', 'bush'),
            ('veggies', 'vegetables'), ('hydrant', 'fire hydrant'), ('bag', 'backpack'), ('tap', 'faucet'),
            ('kid', 'child'), ('signal light', 'traffic light'), ('doughnuts', 'donuts'), ('mom', 'mother'),
            ('palm tree', 'palm'), ('traffic light', 'traffic signal'), ('TV', 'television'), ('girl', 'young girl'),
            ('mountains', 'hills'), ('desk', 'table'), ('bag', 'backpack'), ('table', 'counter'),
            ('computer mice', 'computer mouse')]

verbs = ['both', 'stacked on', 'holding', 'wearing', 'throwing', 'sitting',
         'playing', 'leading', 'sprinkled on', 'riding', 'looking', 'carrying',
         'boarding', 'hitting', 'petting', 'standing', 'lying', 'watching', 'using',
         'reading', 'hanging', 'catching', 'cutting', 'driving', 'using', 'attached',
         'swinging', 'feeding', 'pulling', 'walking', 'reaching', 'parked',
         'reflected', 'leaning', 'driving', 'kicking', 'working', 'sniffing',
         'tossing', 'skating', 'eating', 'about to hit', 'waiting', 'making',
         'entering', 'full of', 'surrounding', 'pushing', 'covering', 'talking',
         'hanging', 'cooking']


def draw_image_details(chosen_image_id, scene_graph_for_image_id, original_qa=None, augmented_qa=None, output_path=None,
                       print_relations=False):
    thickness = 0
    font_scale = 0.7
    ascend = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    lineType = cv2.LINE_AA
    # lineType = cv2.LINE_8
    image_path = osp.join(images_dir, chosen_image_id + '.jpg')
    img_cv = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    colors = list(colors_dict.values())
    for idx, (obj_key, obj_dict) in enumerate(scene_graph_for_image_id['objects'].items()):
        start_point = (obj_dict['x'], obj_dict['y'])
        end_point = (obj_dict['x'] + obj_dict['w'], obj_dict['y'] + obj_dict['h'])
        text_point = (obj_dict['x'], obj_dict['y'] + obj_dict['h'] - ascend)
        color = colors[idx % len(colors)]
        img_rgb = cv2.rectangle(img_rgb, start_point, end_point, color, thickness)
        img_rgb = cv2.putText(img_rgb, obj_dict['name'], text_point, font, fontScale=font_scale,
                              color=color, thickness=thickness,
                              lineType=lineType)
        if print_relations:
            print(obj_dict['name'])
            for relation in obj_dict['relations']:
                print(f"{relation['name']} - {scene_graph_for_image_id['objects'][relation['object']]['name']}")
            print()
    fig = plt.figure(figsize=(20, 9))
    if original_qa and augmented_qa:
        plt.suptitle(f"{original_qa[0]}    {original_qa[1]} -> \n {augmented_qa[0]}    {augmented_qa[1]}", fontsize=26)
    elif original_qa and not augmented_qa:
        plt.suptitle(f"{original_qa[0]}    {original_qa[1]} -> FAIL", fontsize=26)
    plt.imshow(img_rgb, interpolation='nearest')
    if output_path:
        plt.savefig(output_path)
    else:
        print("plt.show()")
        plt.show()
    plt.close(fig)


def words_similarity(a, b):
    seq = difflib.SequenceMatcher(None, a, b)
    return seq.ratio()


from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

plural_bank = ['people', 'scissors', 'men', 'women', 'computer mouses', 'clothes', 'tongs']
descriptors = ['long', 'short', 'medium', 'small']


def isplural(txt):
    txt_words = txt.split(" ")
    if len(txt_words) > 1:
        if txt_words[0] in colors_names or txt_words[0] in descriptors:
            txt = " ".join(txt_words[1:])
    if txt in plural_bank:
        return True
    lemma = wnl.lemmatize(txt, 'n')
    plural = True if txt is not lemma else False

    if not plural:
        word_words = txt.split(" ")
        if len(word_words) > 1:
            second_word = word_words[1]
            lemma_second_word = wnl.lemmatize(second_word, 'n')
            plural = True if second_word is not lemma_second_word else False
    # return plural, lemma
    return plural

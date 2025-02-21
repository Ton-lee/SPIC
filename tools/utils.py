"""useful utils"""
import json
import math
import os
import argparse
import time
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from collections import namedtuple
import numpy as np
from tqdm import trange
import itertools
import cv2
import collections
from scipy.signal import convolve2d as conv
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from collections import Counter


classes_HumanSketch = ['airplane', 'alarm clock', 'angel', 'ant', 'apple', 'arm', 'armchair', 'ashtray', 'axe', 'backpack', 'banana', 'barn', 'baseball bat', 'basket', 'bathtub', 'bear (animal)', 'bed', 'bee', 'beer-mug', 'bell', 'bench', 'bicycle', 'binoculars', 'blimp', 'book', 'bookshelf', 'boomerang', 'bottle opener', 'bowl', 'brain', 'bread', 'bridge', 'bulldozer', 'bus', 'bush', 'butterfly', 'cabinet', 'cactus', 'cake', 'calculator', 'camel', 'camera', 'candle', 'cannon', 'canoe', 'car (sedan)', 'carrot', 'castle', 'cat', 'cell phone', 'chair', 'chandelier', 'church', 'cigarette', 'cloud', 'comb', 'computer monitor', 'computer-mouse', 'couch', 'cow', 'crab', 'crane (machine)', 'crocodile', 'crown', 'cup', 'diamond', 'dog', 'dolphin', 'donut', 'door', 'door handle', 'dragon', 'duck', 'ear', 'elephant', 'envelope', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fire hydrant', 'fish', 'flashlight', 'floor lamp', 'flower with stem', 'flying bird', 'flying saucer', 'foot', 'fork', 'frog', 'frying-pan', 'giraffe', 'grapes', 'grenade', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'head', 'head-phones', 'hedgehog', 'helicopter', 'helmet', 'horse', 'hot air balloon', 'hot-dog', 'hourglass', 'house', 'human-skeleton', 'ice-cream-cone', 'ipod', 'kangaroo', 'key', 'keyboard', 'knife', 'ladder', 'laptop', 'leaf', 'lightbulb', 'lighter', 'lion', 'lobster', 'loudspeaker', 'mailbox', 'megaphone', 'mermaid', 'microphone', 'microscope', 'monkey', 'moon', 'mosquito', 'motorbike', 'mouse (animal)', 'mouth', 'mug', 'mushroom', 'nose', 'octopus', 'owl', 'palm tree', 'panda', 'paper clip', 'parachute', 'parking meter', 'parrot', 'pear', 'pen', 'penguin', 'person sitting', 'person walking', 'piano', 'pickup truck', 'pig', 'pigeon', 'pineapple', 'pipe (for smoking)', 'pizza', 'potted plant', 'power outlet', 'present', 'pretzel', 'pumpkin', 'purse', 'rabbit', 'race car', 'radio', 'rainbow', 'revolver', 'rifle', 'rollerblades', 'rooster', 'sailboat', 'santa claus', 'satellite', 'satellite dish', 'saxophone', 'scissors', 'scorpion', 'screwdriver', 'sea turtle', 'seagull', 'shark', 'sheep', 'ship', 'shoe', 'shovel', 'skateboard', 'skull', 'skyscraper', 'snail', 'snake', 'snowboard', 'snowman', 'socks', 'space shuttle', 'speed-boat', 'spider', 'sponge bob', 'spoon', 'squirrel', 'standing bird', 'stapler', 'strawberry', 'streetlight', 'submarine', 'suitcase', 'sun', 'suv', 'swan', 'sword', 'syringe', 't-shirt', 'table', 'tablelamp', 'teacup', 'teapot', 'teddy-bear', 'telephone', 'tennis-racket', 'tent', 'tiger', 'tire', 'toilet', 'tomato', 'tooth', 'toothbrush', 'tractor', 'traffic light', 'train', 'tree', 'trombone', 'trousers', 'truck', 'trumpet', 'tv', 'umbrella', 'van', 'vase', 'violin', 'walkie talkie', 'wheel', 'wheelbarrow', 'windmill', 'wine-bottle', 'wineglass', 'wrist-watch', 'zebra']
classes_QuickDraw = ['The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling fan', 'cell phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup', 'diamond', 'dishwasher', 'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 'flower', 'flying saucer', 'foot', 'fork', 'frog', 'frying pan', 'garden', 'garden hose', 'giraffe', 'goatee', 'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck', 'hockey stick', 'horse', 'hospital', 'hot air balloon', 'hot dog', 'hot tub', 'hourglass', 'house', 'house plant', 'hurricane', 'ice cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint can', 'paintbrush', 'palm tree', 'panda', 'pants', 'paper clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup truck', 'picture frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote control', 'rhinoceros', 'rifle', 'river', 'roller coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school bus', 'scissors', 'scorpion', 'screwdriver', 'sea turtle', 'see saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping bag', 'smiley face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop sign', 'stove', 'strawberry', 'streetlight', 'string bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing set', 'sword', 'syringe', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis racquet', 'tent', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine bottle', 'wine glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']
classes_ImageNet = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01531178', 'n01537544', 'n01560419', 'n01582220', 'n01592084', 'n01601694', 'n01608432', 'n01614925', 'n01622779', 'n01630670', 'n01632458', 'n01632777', 'n01644900', 'n01664065', 'n01665541', 'n01667114', 'n01667778', 'n01675722', 'n01677366', 'n01685808', 'n01687978', 'n01693334', 'n01695060', 'n01698640', 'n01728572', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01739381', 'n01740131', 'n01742172', 'n01749939', 'n01751748', 'n01753488', 'n01755581', 'n01756291', 'n01770081', 'n01770393', 'n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750', 'n01775062', 'n01776313', 'n01795545', 'n01796340', 'n01798484', 'n01806143', 'n01818515', 'n01819313', 'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01860187', 'n01877812', 'n01883070', 'n01910747', 'n01914609', 'n01924916', 'n01930112', 'n01943899', 'n01944390', 'n01950731', 'n01955084', 'n01968897', 'n01978287', 'n01978455', 'n01984695', 'n01985128', 'n01986214', 'n02002556', 'n02006656', 'n02007558', 'n02011460', 'n02012849', 'n02013706', 'n02018207', 'n02018795', 'n02027492', 'n02028035', 'n02037110', 'n02051845', 'n02058221', 'n02077923']
classes_Caltech = ['BACKGROUND_Google', 'Faces', 'Faces_easy', 'Leopards', 'Motorbikes', 'accordion', 'airplanes', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']
classes_Cifar = ['%02d' % i for i in range(100)]
classes_Fasion = ['%d' % i for i in range(10)]


def get_file_pairs(root: str, save_path: str, split: str):
    """
    generate file pair list and save to target path
    :param root: root directory
    :param save_path: save the pair file to the target path, named 'train_pairs.txt' or 'test_pairs.txt'
    :param split: 'train' or 'test'
    :return: whether saved successfully
    """
    # format:
    #   train: image_path sketch_path(relative path to root)
    #   test: image_path
    if split == 'train':
        path_img = os.path.join(root, 'images', 'train')
        path_sketch = os.path.join(root, 'sketches', 'train')
        paths_img = [os.path.join('images', 'train', p) for p in os.listdir(path_img) if p[-4:] == '.jpg']
        paths_sketch = [os.path.join('sketches', 'train', p) for p in os.listdir(path_sketch) if p[-6] == '0']
        paths_img.sort()
        paths_sketch.sort()
        with open(save_path, 'w') as f:
            for i in range(len(paths_img)):
                image_path = paths_img[i]
                sketch_path = paths_sketch[i]
                f.write('{} {}\n'.format(image_path, sketch_path))
        return True
    elif split == 'trainval':
        path_img = os.path.join(root, 'images', 'train')
        path_sketch = os.path.join(root, 'sketches', 'train')
        paths_img1 = [os.path.join('images', 'train', p) for p in os.listdir(path_img) if p[-4:] == '.jpg']
        paths_sketch1 = [os.path.join('sketches', 'train', p) for p in os.listdir(path_sketch) if p[-6] == '0']
        path_img = os.path.join(root, 'images', 'val')
        path_sketch = os.path.join(root, 'sketches', 'val')
        paths_img2 = [os.path.join('images', 'val', p) for p in os.listdir(path_img) if p[-4:] == '.jpg']
        paths_sketch2 = [os.path.join('sketches', 'val', p) for p in os.listdir(path_sketch) if p[-6] == '0']
        paths_img = paths_img1 + paths_img2
        paths_sketch = paths_sketch1 + paths_sketch2
        paths_img.sort()
        paths_sketch.sort()
        with open(save_path, 'w') as f:
            for i in range(len(paths_img)):
                image_path = paths_img[i]
                sketch_path = paths_sketch[i]
                f.write('{} {}\n'.format(image_path, sketch_path))
        return True
    else:
        assert split == 'test'
        path_img = os.path.join(root, 'images', 'test')
        paths_img = [os.path.join('images', 'test', p) for p in os.listdir(path_img) if p[-4:] == '.jpg']
        with open(save_path, 'w') as f:
            for i in range(len(paths_img)):
                image_path = paths_img[i]
                f.write('{}\n'.format(image_path))
        return True


def get_file_pairs_edges(root: str, save_path: str, split: str):
    """
    generate file pair list (of images and edges) and save to target path
    :param root: root directory
    :param save_path: save the pair file to the target path, named 'train_pairs.txt' or 'test_pairs.txt'
    :param split: 'train' or 'test'
    :return: whether saved successfully
    """
    # format:
    #   train: image_path sketch_path(relative path to root)
    #   test: image_path
    if split == 'train':
        path_img = os.path.join(root, 'images', 'train')
        path_edge = os.path.join(root, 'edges-single', 'train')
        paths_img = [os.path.join('images', 'train', p) for p in os.listdir(path_img) if p[-4:] == '.jpg']
        paths_edge = [os.path.join('sketches', 'train', p) for p in os.listdir(path_edge)]
        paths_img.sort()
        paths_edge.sort()
        with open(save_path, 'w') as f:
            for i in range(len(paths_img)):
                image_path = paths_img[i]
                edge_path = paths_edge[i]
                f.write('{} {}\n'.format(image_path, edge_path))
        return True
    elif split == 'trainval':
        path_img = os.path.join(root, 'images', 'train')
        path_edge = os.path.join(root, 'edges-single', 'train')
        paths_img1 = [os.path.join('images', 'train', p) for p in os.listdir(path_img) if p[-4:] == '.jpg']
        paths_edge1 = [os.path.join('edges-single', 'train', p) for p in os.listdir(path_edge)]
        path_img = os.path.join(root, 'images', 'val')
        path_edge = os.path.join(root, 'edges-single', 'val')
        paths_img2 = [os.path.join('images', 'val', p) for p in os.listdir(path_img) if p[-4:] == '.jpg']
        paths_edge2 = [os.path.join('edges-single', 'val', p) for p in os.listdir(path_edge)]
        paths_img = paths_img1 + paths_img2
        paths_edge = paths_edge1 + paths_edge2
        paths_img.sort()
        paths_edge.sort()
        with open(save_path, 'w') as f:
            for i in range(len(paths_img)):
                image_path = paths_img[i]
                sketch_path = paths_edge[i]
                f.write('{} {}\n'.format(image_path, sketch_path))
        return True
    else:
        assert split == 'test'
        path_img = os.path.join(root, 'images', 'test')
        paths_img = [os.path.join('images', 'test', p) for p in os.listdir(path_img) if p[-4:] == '.jpg']
        with open(save_path, 'w') as f:
            for i in range(len(paths_img)):
                image_path = paths_img[i]
                f.write('{}\n'.format(image_path))
        return True


def read_json(path: str):
    """
    read json file
    :param path: path of a json file
    :return: a directory
    """
    with open(path, 'r') as f:
        s = json.load(f)
    return s


def arg_parse():
    """
parse arguments from the terminal
    """
    parser = argparse.ArgumentParser(description='Sketch Extracting Networks')

    parser.add_argument('--savedir', type=str, default='results/',
                        help='path to save result and checkpoint')
    parser.add_argument('--datadir', type=str, default='/home/Users/dqy/Dataset/BSDS500/',
                        help='dir to the dataset')
    parser.add_argument('--ablation', action='store_true',
                        help='not use bsds val set for training')
    parser.add_argument('--dataset', type=str, default='BSDS',
                        help='data settings for BSDS, Multicue and NYUD datasets')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed (default: None)')
    parser.add_argument('--gpu', type=str, default='',
                        help='gpus available')
    parser.add_argument('--checkinfo', action='store_true',
                        help='only check the information about the model: model size, flops')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of total epochs to run')
    parser.add_argument('--iter-size', type=int, default=24,
                        help='number of samples in each iteration')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='number of samples in each batch')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='initial learning rate for all weights')
    parser.add_argument('--lr-type', type=str, default='multistep',
                        help='learning rate strategy [cosine, multistep]')
    parser.add_argument('--lr-steps', type=str, default='10-16',
                        help='steps for multistep learning rate')
    parser.add_argument('--opt', type=str, default='adam',
                        help='optimizer')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay for all weights')
    parser.add_argument('-j', '--workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--lmbda', type=float, default=1.1,
                        help='weight on negative pixels')
    parser.add_argument('--resume', action='store_true',
                        help='use latest checkpoint if have any')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save-freq', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--evaluate', type=str, default=None,
                        help='full path to checkpoint to be evaluated')
    parser.add_argument('--evaluate-converted', action='store_true',
                        help='convert the checkpoint to vanilla cnn, then evaluate')
    parser.add_argument('--weight_connection', type=float, default=1,
                        help='weight on loss of connection predictions')
    parser.add_argument('--weight_entropy', type=float, default=1,
                        help='weight on loss of entropy predictions')
    parser.add_argument('--weight_neighbor', type=float, default=1,
                        help='weight on loss of neighbor strength')
    parser.add_argument('--model', type=str, default='Sketcher')
    parser.add_argument('--config_file', type=str, default='config/BSDS.yaml')
    parser.add_argument('--classes', type=str, default='',
                        help='valid for classification, limit the classes to consider')
    parser.add_argument('--class_count', type=int, default=5,
                        help='valid when classes is random for classification, the count of classes to process')
    parser.add_argument('--gt', type=str, default='sketch')
    parser.add_argument('--freeze_Pidinet', action='store_true',
                        help='freeze Pidinet of the pretrained')
    parser.add_argument('--freeze_feature', action='store_true',
                        help='freeze feature extractor of the pretrained')
    parser.add_argument('--append_image', action='store_true',
                        help='if append the original image in the test result')
    parser.add_argument('--append_nms', action='store_true',
                        help='if append the nms processed edge in the test result')
    parser.add_argument('--pair_name', type=str, default='None')
    parser.add_argument('--test_on_train', action='store_true', help='if to test on the training'
                                                                     'dataset')
    parser.add_argument('--save_result', action='store_true', help='if to save the result as images')
    parser.add_argument('--weight_method', type=str, default='none')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args


def arg_refine(args):
    """
refine arguments
    """
    if args.seed is None:
        args.seed = int(time.time())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.use_cuda = torch.cuda.is_available()

    if args.lr_steps is not None and not isinstance(args.lr_steps, list):
        args.lr_steps = list(map(int, args.lr_steps.split('-')))

    dataset_setting_choices = ['BSDS', 'Custom', 'BSDS-HED', 'BSDS-Sketch', 'BSDS-HED-NMS', 'BSDS-KPT', 'BSDS-self',
                               'BSDS-SO', 'BSDS-DA', 'BSDS-SA',
                               'BSDS-KPT-SO', 'BSDS-KPT-SA',
                               'BSDS-Edge',
                               'HumanSketch', 'QuickDraw',
                               'ImageNet', 'Caltech',
                               'ImageNet_Canny', 'Caltech_Canny',
                               'ImageNet_PidiNet', 'Caltech_PidiNet',
                               'Cifar', 'Fasion',
                               'Cifar_Canny', 'Fasion_Canny',
                               'Cifar_PidiNet', 'Fasion_PidiNet']
    if not isinstance(args.dataset, list):
        assert args.dataset in dataset_setting_choices, 'unrecognized data setting %s, please choose from %s' % (
            str(args.dataset), str(dataset_setting_choices))
        # args.dataset = list(args.dataset.strip().split('-'))
    if args.weight_method not in ['none', 'sample', 'element']:
        args.weight_method = 'none'
        print('Invalid weight method. Use none instead.')
    if args.pair_name == 'None':
        args.pair_name = None
    print(args)
    return args


def get_model_parm_nums(model):
    total = sum([param.numel() for param in model.parameters()])
    total = float(total) / 1e6
    return total


def load_checkpoint(args, running_file):
    model_dir = os.path.join(args.savedir, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = ''

    if args.evaluate is not None:
        model_filename = args.evaluate
    else:
        if os.path.exists(latest_filename):
            with open(latest_filename, 'r') as fin:
                model_filename = fin.readlines()[0].strip()
    loadinfo = "=> loading checkpoint from '{}'".format(model_filename)
    print(loadinfo)

    state = None
    if os.path.exists(model_filename):
        state = torch.load(model_filename, map_location='cpu')
        loadinfo2 = "=> loaded checkpoint '{}' successfully".format(model_filename)
    else:
        loadinfo2 = "no checkpoint loaded"
    print(loadinfo2)
    running_file.write('%s\n%s\n' % (loadinfo, loadinfo2))
    running_file.flush()

    return state


def adjust_learning_rate(optimizer, epoch, args):
    method = args.lr_type
    lr = -1
    if method == 'cosine':
        T_total = float(args.epochs)
        T_cur = float(epoch)
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        lr = args.lr
        for epoch_step in args.lr_steps:
            if epoch >= epoch_step:
                lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    str_lr = '%.6f' % lr
    return str_lr


def save_checkpoint(state, epoch, root, saveID, keep_freq=10):
    filename = 'checkpoint_%03d.pth' % epoch
    model_dir = os.path.join(root, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # write new checkpoint
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    print("=> saved checkpoint '{}'".format(model_filename))

    # remove old model
    if saveID is not None and (saveID + 1) % keep_freq != 0:
        filename = 'checkpoint_%03d.pth' % saveID
        model_filename = os.path.join(model_dir, filename)
        if os.path.exists(model_filename):
            os.remove(model_filename)
            print('=> removed checkpoint %s' % model_filename)

    print('##########Time##########', time.strftime('%Y-%m-%d %H:%M:%S'))
    return epoch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def cross_entropy_loss_RCF(prediction, labelf, beta):
    prediction = torch.clip(prediction, 0, 1)
    labelf = labelf[:, None, :, :].repeat(1, prediction.shape[1], 1, 1)  # recent: align to prediction
    label = labelf.long()
    mask = labelf.clone()
    num_positive = torch.sum(label == 1).float()
    num_negative = torch.sum(label == 0).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)  # ratio > th
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)  # ratio == 0
    mask[label == 2] = 0  # 0 < ratio < th
    try:
        assert prediction.shape == labelf.shape
    except AssertionError as e:
        print(e)
        raise AssertionError(e)
    cost = F.binary_cross_entropy(
        prediction, labelf, weight=mask, reduction='sum')

    return cost


def cross_entropy_loss_neighbor(prediction, labelf, beta):
    prediction = torch.clip(prediction, 0, 1)
    label = labelf.long()
    num_positive = torch.sum(label == 1).float()
    num_negative = torch.sum(label == 0).float()
    weight_positive = 1.0 * num_negative / (num_positive + num_negative)
    weight_negative = beta * num_positive / (num_positive + num_negative)
    log_ = torch.log(prediction + torch.Tensor([1e-30]).to(prediction.device))
    weight_log = torch.ones_like(log_)
    weight_log[label == 0] = 1e-30
    weight_log[label == 2] = 1e-30
    log_ *= weight_log
    log_minus = torch.log(1 - prediction + torch.Tensor([1e-30]).to(prediction.device))
    loss_negative = -torch.sum((label == 0) * log_minus)  # 负样本，仍然采用交叉熵
    loss_positive = -torch.sum(torch.max_pool2d(log_,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=(1, 1)) *
                               (label == 1))
    cost = loss_negative * weight_negative + loss_positive * weight_positive
    return cost, loss_positive, loss_negative, weight_positive, weight_negative


def cross_entropy_loss_RCF_points(prediction, labelf, beta, indexes):
    """
对于连接的分类，采用二分类问题的交叉熵损失
    :param prediction: 预测的连接矩阵(1,N,N)
    :param labelf: 连接矩阵实际值(1,N_gt,N_gt)
    :param beta: 负样本（无连接）部分的权重
    :return: 各点交叉熵之和
    """
    label = labelf.long()
    mask = labelf.clone()
    num_positive = torch.sum(label == 1).float()
    num_negative = torch.sum(label == 0).float()
    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    # print(prediction.shape)
    # print(prediction[:, indexes, :][:, :, indexes].shape, labelf.shape)
    cost = F.binary_cross_entropy(
        prediction[:, indexes, :][:, :, indexes], labelf, weight=mask, reduction='sum')  # 只对有效部分计算损失
    return cost


def cross_entropy_loss_points(prediction, beta, positions):
    """
对于连接的分类，采用二分类问题的交叉熵损失
    :param prediction: 预测的点强度, (B,1,H,W)
    :param positions: 实际需要预测到的坐标位置, (B,N,2)
    :param beta: 负样本（无连接）部分的权重
    :return: 各点交叉熵之和
    """
    label = torch.zeros_like(prediction)
    mask = torch.zeros_like(prediction)
    for b in range(prediction.shape[0]):
        for index in range(positions.shape[-2]):
            label[b, 0, int(positions[b, index, 0]), int(positions[b, index, 1])] = 1
    num_positive = positions.shape[-2]
    num_negative = prediction.shape[-2] * prediction.shape[-1] - num_positive
    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    cost = F.binary_cross_entropy(
        prediction, label, weight=mask, reduction='mean')
    return cost


def cross_entropy_loss_RCF_tri(prediction, labelf, beta, indexes):
    """
对于连接的分类，采用二分类问题的交叉熵损失
    :param prediction: 预测的连接矩阵(1,N,N)，上三角阵，即只有 i<j的位置有值
    :param labelf: 连接矩阵实际值(1,N_gt,N_gt)，上三角阵
    :param beta: 负样本（无连接）部分的权重
    :param indexes: 有效考察样本的序号
    :return: 各点交叉熵之和
    """
    N_gt = labelf.shape[-1]
    label = labelf.long().to(prediction.device)
    mask = labelf.clone().to(prediction.device)
    num_positive = torch.sum(label == 1).float()
    num_negative = torch.sum(label == 0).float() - (N_gt - 1) ** 2 / 2 - N_gt  # 只考虑上三角部分的0标签数量
    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    mask = torch.triu(mask)  # 对下三角部分全部置为0，对角线虽然有权重，但全部预测正确（均为0），对梯度传播没有影响
    # print(prediction.shape)
    # print(prediction[:, indexes, :][:, :, indexes].shape, labelf.shape)
    cost = F.binary_cross_entropy(
        prediction[:, indexes, :][:, :, indexes], labelf, weight=mask, reduction='mean')  # 只对有效部分计算损失
    return cost


def cross_entropy_loss_confidence(prediction, indexes, beta):
    """
对于置信度的分类，采用二分类问题的交叉熵损失
    :param prediction: 预测的置信度(1,N)
    :param beta: 负样本（非目标点）部分的权重
    :param indexes: 目标点的序号
    :return: 各点交叉熵之和
    """
    gt = torch.zeros_like(prediction).to(prediction.device)
    gt[0, indexes] = 1
    mask = torch.zeros_like(prediction).to(prediction.device)
    # print(mask.shape)
    # print(indexes)
    # try:
    #     for index in indexes:
    #         mask[0, index] = 1
    # except RuntimeError as e:
    #     print(e)
    #     print(index)
    #     raise RuntimeError(e)
    mask[0, indexes] = 1
    num_positive = torch.sum(gt == 1).float()
    num_negative = torch.sum(gt == 0).float()
    mask[gt == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[gt == 0] = beta * num_positive / (num_positive + num_negative)
    cost = F.binary_cross_entropy(prediction, gt, weight=mask, reduction='mean')
    return cost


def MSE_loss_proposal(prediction, groundtruth, N_gt, size):
    """
借鉴目标检测的proposal损失函数设计，对每个groundtruth，选择距离其最近的预测坐标，计算欧氏距离
    :param prediction: 预测的点坐标(1,N,2)
    :param groundtruth: 实际点坐标(1,N_gt,2)
    :param N_gt: 实际的点数量
    :param size: 图像尺寸，(H,W)
    """
    indexes = []
    groundtruth = groundtruth / (torch.Tensor(size)[None, None, :]).to(groundtruth.device) * 2 - 1
    for i in range(N_gt):
        gt = groundtruth[:, i, :]
        index = torch.argmin(torch.sum(torch.square(prediction - gt[:, None, :]), dim=2))
        indexes.append(int(index))
    # print(prediction[:, indexes, :].shape, groundtruth.shape)
    return F.mse_loss(prediction[:, indexes, :], groundtruth, reduction='mean'), indexes


def MSE_loss(prediction, groundtruth, weighted='none', input_image=None):
    if weighted == 'none':
        return F.mse_loss(prediction, groundtruth, reduction='mean')
    assert input_image is not None
    if weighted == 'sample':  # 按样本加权
        return F.mse_loss(prediction, groundtruth, reduction='sum') / torch.sum(input_image > 0)
    if weighted == 'element':  # 元素加权
        mask = (input_image > 0).float()
        mask[groundtruth > 0] = 1 - torch.sum(groundtruth > 0) / torch.sum(input_image > 0)
        mask[(groundtruth == 0) * (input_image > 0)] = torch.sum(groundtruth > 0) / \
                                                       torch.sum(input_image > 0)
        return torch.sum(torch.square(prediction - groundtruth) * mask) / torch.sum(input_image > 0)


def entropy_loss(entropy: torch.Tensor, indexes):
    """
计算熵损失，使indexes中的熵尽量小，以外的熵尽量大
    :param entropy: 预测熵值 (1,N)
    :param indexes: 合法点的序号 (N_gt)
    """
    symbol = -torch.ones_like(entropy)
    symbol[:, indexes] = 1
    return torch.mean(entropy * symbol)


def CE_loss(prediction, groundtruth):
    return F.cross_entropy(prediction, groundtruth)


def BCE_loss(prediction, groundtruth, mask='none', reduction='mean'):
    """【仅对连接关系检测适用】去除对角线的交叉熵"""
    mask_ = torch.ones_like(prediction)
    N = prediction.shape[1]
    if mask == 'diagnose':
        mask_ -= torch.eye(N, N).to(prediction.device)
    elif mask == 'weighted':
        ratio = torch.sum(groundtruth) / N / (N-1)  # 正样本的比例
        if ratio not in [0, 1]:
            mask_[groundtruth == 1] = 1 / ratio
            mask_[groundtruth == 0] = 1 / (1-ratio)
            mask_ -= torch.eye(N, N).to(prediction.device) / (1-ratio)  # 将对角线置为0
        else:
            mask_ -= torch.eye(N, N).to(prediction.device)
    return F.binary_cross_entropy(prediction, groundtruth, mask_, reduction=reduction)


def Range_loss(prediction: torch.Tensor):
    connection_sum = torch.sum(prediction, dim=2)
    over_range = F.relu(connection_sum - torch.Tensor([4]).to(prediction.device)) + \
            F.relu(torch.Tensor([1]).to(prediction.device) - connection_sum)
    return torch.mean((over_range > 0).to(torch.float))


def masked_CE(prediction, groundtruth, mask, weights=1):
    """【仅对关键点检测适用】给定权重的交叉熵"""
    if weights == -1:
        weights = 1 / np.array([0.01353759, 0.09537743, 0.00505805, 0.88602693])
    loss = -torch.sum((groundtruth * torch.log(prediction + 1e-30)
                       + (1 - groundtruth) * torch.log(1 - prediction + 1e-30)) * mask * torch.Tensor(weights).reshape(
        [1, 4, 1, 1]).to(mask.device)) / torch.sum(mask)
    if torch.isnan(loss):
        print('illegal loss!')
    return loss


def save_json(path, data):
    dict_data = json.dumps(data)  # 转化为json格式文件
    with open(path, 'w') as f:
        f.write(dict_data)


Sketch = namedtuple('Sketch', ['positions', 'connections', 'confidence'])
SketchMeta = namedtuple('SketchMeta', ['edge',
                                       'image_keypoint_nms',
                                       'flag_cross',
                                       'flag_corner',
                                       'flag_end',
                                       'flag_keypoints',
                                       'pos_keypoint'])


def plot_sketch(positions, size, connections=None, entropy=None, threshold_point=None, threshold_connection=None, base=None):
    """
    根据所给点和连接关系绘制素描图
    :param positions: 2*N 或 N*2 坐标数组
    :param size: 图像尺寸
    :param connections: N*N 的连接关系矩阵
    :param entropy: 各个点的熵
    :param threshold_point: 熵的阈值，高于阈值的点将被忽略
    :param threshold_connection: 连接关系的阈值，低于阈值强度的连接关系将被忽略
    :param base: 底图，将以 1/5 的强度放置在素描线下
    :return: 白色像素为连接线，红色为素描点
    """
    assert size.__class__ in [list, tuple] and len(size) == 2
    size = list(size)
    if base is not None:
        sketch = (base / 5).astype('uint8')
    else:
        sketch = np.zeros(size + [3], 'uint8')
    sketch_pairs = np.ones(size + [2], tuple) * -1
    plot_times = np.zeros(size)
    positions = np.array(positions)
    try:
        assert positions.shape[1] == 2
    except AssertionError:
        assert positions.shape[0] == 2
        positions = positions.transpose()
    if positions.shape[0] == 0:
        return sketch
    if connections is not None:
        assert connections.__class__ == np.ndarray and connections.shape[0] == connections.shape[1] == positions.shape[
            0]
    if entropy is None:
        entropy = np.zeros([positions.shape[0]])
    assert len(entropy) == positions.shape[0] or entropy is None
    if threshold_point is None:
        threshold_point = np.inf
    if threshold_connection is None:
        threshold_connection = 0
    if np.max(positions) <= 2:  # ranging in (-1,1)
        positions = np.floor((positions + 1) / 2 * np.array(size)[None])
    positions[positions[:, 0] >= size[0], 0] = size[0] - 1
    positions[positions[:, 1] >= size[1], 1] = size[1] - 1
    if connections is None:  # only plot the points
        for i in range(positions.shape[0]):
            sketch[int(positions[i, 0]), int(positions[i, 1]), :] = [255, 0, 0]
        # print(positions[:5, :])
        return sketch
    for i in range(positions.shape[0]):
        if entropy[i] >= threshold_point:
            continue
        for j in range(i + 1, positions.shape[0]):
            if entropy[j] >= threshold_point:
                continue
            if connections[i, j] <= threshold_connection:
                continue
            min_line, max_line = min(positions[i, 0], positions[j, 0]), max(positions[i, 0], positions[j, 0])
            min_column, max_column = min(positions[i, 1], positions[j, 1]), max(positions[i, 1], positions[j, 1])
            length = int(max(max_line - min_line, max_column - min_column)) + 1
            fill_line = np.round(np.linspace(positions[i, 0], positions[j, 0], length))
            fill_column = np.round(np.linspace(positions[i, 1], positions[j, 1], length))
            strength = int(connections[i, j]*255)
            for index in range(length):
                sketch[int(fill_line[index]), int(fill_column[index]), :] = [max(sketch[int(fill_line[index]), int(fill_column[index]), 0], strength)] * 3  # 防止小值覆盖大值
                plot_times[int(fill_line[index]), int(fill_column[index])] += 1
                if sketch_pairs[int(fill_line[index]), int(fill_column[index]), 0] == -1:
                    sketch_pairs[int(fill_line[index]), int(fill_column[index]), 0] = [i, ]
                    sketch_pairs[int(fill_line[index]), int(fill_column[index]), 1] = [j, ]
                else:
                    sketch_pairs[int(fill_line[index]), int(fill_column[index]), 0].append(i)
                    sketch_pairs[int(fill_line[index]), int(fill_column[index]), 1].append(j)
            # sketch[int(positions[i, 0]), int(positions[i, 1]), :] = [255, 0, 0]
            # sketch[int(positions[j, 0]), int(positions[j, 1]), :] = [255, 0, 0]
    for i in range(positions.shape[0]):  # 再次强调关键点
        sketch[int(positions[i, 0]), int(positions[i, 1]), :] = [255, 0, 0]
    return sketch, sketch_pairs, plot_times


def plot_sketch_tensor(positions, size, connections, threshold=0.1):
    edge = torch.zeros(size).to(positions.device)
    N = len(positions)
    positions = positions.to(torch.int)
    for i in range(N):
        for j in range(i+1, N):
            if connections[i, j] < threshold:
                continue
            line1 = positions[i, 0]
            line2 = positions[j, 0]
            column1 = positions[i, 1]
            column2 = positions[j, 1]
            L = max(torch.abs(line1 - line2) + 1, torch.abs(column1 - column2) + 1)
            lines = torch.linspace(line1, line2, L).to(torch.long)
            columns = torch.linspace(column1, column2, L).to(torch.long)
            edge[lines, columns] = connections[i, j]
    return edge


def generate_kernels():
    a = [0, 1, 2, 3, 5, 6, 7, 8]
    choice_list = list(itertools.combinations(a, 2))
    kernels_pos = torch.ones([len(choice_list), 1, 3, 3])
    kernels_neg = torch.zeros([len(choice_list), 1, 3, 3])
    for choice_index in range(len(choice_list)):
        choice = choice_list[choice_index]
        kernels_pos[choice_index, 0, 1, 1] = 0
        for index in choice:
            line, column = index // 3, index % 3
            kernels_pos[choice_index, 0, line, column] = 0
            kernels_neg[choice_index, 0, line, column] = 1
    return kernels_pos, kernels_neg


def rotate(image, angle):
    # 获取图像的尺寸，然后确定中心
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # 获取旋转矩阵（应用角度逆时针旋转），然后抓住正弦和余弦（即矩阵的旋转分量）
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # 计算图像新的边界尺寸
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))


def get_clip_range(shape_ori, angle):
    h, w = shape_ori
    cos = np.abs(np.cos(angle / 180 * np.pi))
    sin = np.abs(np.sin(angle / 180 * np.pi))
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    h_, w_ = h, w
    if angle in [45.0, 135.0]:
        w_, h_ = int(min(h, w) / np.sqrt(2)), int(min(h, w) / np.sqrt(2))
    elif angle in [22.5, 157.5]:
        w_ = int((h * sin - w * cos) / (sin ** 2 - cos ** 2))
        h_ = int((h * cos - w * sin) / (cos ** 2 - sin ** 2))
    elif angle in [67.5, 112.5]:
        if h > w:
            w_ = w
            h_ = int((w - w_ * cos) / sin)
        else:
            h_ = h
            w_ = int((h - h_ * cos) / sin)
    elif angle == 90:
        w_, h_ = min(h, w), min(h, w)
    min_line, max_line = int(nH / 2 - h_ / 2), int(nH / 2 + h_ / 2)
    min_column, max_column = int(nW / 2 - w_ / 2), int(nW / 2 + w_ / 2)
    return min_line, max_line, min_column, max_column


def rotate_image(image, angle):
    image_full = rotate(image, angle)
    nH, nW, _ = image_full.shape
    h, w, _ = image.shape
    min_l, max_l, min_c, max_c = get_clip_range(image.shape[:2], angle)
    return image_full[min_l:max_l, min_c:max_c, :]


def filter_neighbor(pos):
    """
    对旋转后的边缘点坐标进行滤波，避免锯齿效应
    """
    pos_filtered = pos.copy()
    dist_line = np.abs(pos[:, 0:1] - pos[:, 0:1].T)
    dist_column = np.abs(pos[:, 1:2] - pos[:, 1:2].T)
    valid_pairs = (dist_line < 1) * (dist_column < 1)
    for i in range(pos.shape[0]):
        valid_neighbor = np.where(valid_pairs[:, i] == 1)
        neighbor_pos = [pos[n, :] for n in valid_neighbor][0]
        span_line, span_column = np.max(neighbor_pos[:, 0]) - np.min(neighbor_pos[:, 0]), \
                                 np.max(neighbor_pos[:, 1]) - np.min(neighbor_pos[:, 1])
        if span_line > span_column:
            pos_filtered[i, 1] = np.mean(neighbor_pos[:, 1])
        else:
            pos_filtered[i, 0] = np.mean(neighbor_pos[:, 0])
    return pos_filtered


def get_index(line, column, h, w, neighbor_range):
    """
获取邻域范围
    """
    # min_i = max(0, line - neighbor_range)
    min_i = line - neighbor_range if line > neighbor_range else 0
    # max_i = min(h - 1, line + neighbor_range)
    max_i = line + neighbor_range if line < h - neighbor_range else h - 1
    # min_j = max(0, column - neighbor_range)
    min_j = column - neighbor_range if column > neighbor_range else 0
    # max_j = min(w - 1, column + neighbor_range)
    max_j = column + neighbor_range if column < w - neighbor_range else w - 1
    return min_i, max_i, min_j, max_j


def branch_count_2path(valid, neighbor_range=8, return_branch_map=True):  # 基于两分支的连通数计算
    """
基于两分支方法计算连通分支数。
    :param valid: 有效点，需要计算连通分支
    :param neighbor_range: 4邻域或8邻域搜索
    :param return_branch_map: 是否需要返回分支序号标记
    :return: 连通分支数，或连通分支数和分支序号标记
    """
    edge = valid.copy().astype('int')
    idx = 2  # 从2开始标注，避免与边缘的标记1混淆
    equal_pairs = []
    h, w = valid.shape
    for i in range(h):
        min_i = i - 1 if i > 0 else 0
        for j in range(w):
            if edge[i, j] == 0 or edge[i, j] == -1:  # 非边缘部分
                continue
            min_j = j - 1 if j > 0 else 0
            max_j = j if j == w - 1 else j + 1
            neighbor = edge[min_i: i + 1, min_j:max_j + 1] if neighbor_range == 8 else edge[
                [min_i, i], [j, min_j]]  # 8邻域，考察其上方的6个点；4邻域，考察其上方和左方的两个点
            ids_previous = list(set(neighbor[neighbor > 1]))  # 上方标记过的序号，取>1的数值
            n_previous = ids_previous.__len__()
            if n_previous == 0:
                edge[i, j] = idx
                idx += 1  # continue here
                continue
            if n_previous == 1:
                edge[i, j] = ids_previous[0]  # continue here
                continue
            if n_previous > 3:
                continue
            if n_previous > 1:
                edge[i, j] = min(ids_previous)
                if ids_previous not in equal_pairs:
                    if n_previous == 3:
                        equal_pairs.append(ids_previous[:2])
                        equal_pairs.append(ids_previous[1:])
                        continue
                    else:
                        equal_pairs.append(ids_previous)  # continue here
                        continue
    eq_marks, eq_label, eq_count = np.zeros(idx - 2).astype('int'), 1, 0
    # 无等价类，则直接获得分支数和分支图
    if equal_pairs.__len__() == 0:
        if not return_branch_map:
            return idx - 2
        edge[edge > 0] -= 1
        edge[edge == -1] = 0
        return idx - 2, edge.astype('int')
    # 有等价类，需要进行等价类合并
    for i in range(len(equal_pairs)):
        i1, i2 = equal_pairs[i]
        i1, i2 = i1 - 2, i2 - 2
        m1, m2 = eq_marks[i1], eq_marks[i2]
        if m1 == 0:
            if m2 == 0:
                eq_marks[[i1, i2]] = eq_label
                eq_label += 1  # 等价类标签+1
                eq_count += 1  # 等价类数量+1
            else:
                eq_marks[i1] = m2
                eq_count += 1  # 等价类数量+1
        else:
            if m2 == 0:
                eq_marks[i2] = m1
                eq_count += 1  # 等价类数量+1
            else:
                if m1 != m2:
                    eq_marks[eq_marks == m2] = m1  # 等价类合并
                    eq_count += 1  # 等价类数量+1
    if not return_branch_map:
        return idx - 2 - eq_count
    # 根据合并后的等价类获得分支图
    label = 1
    branch_id_ori, branch_id_pre = [], []
    edge_merged = np.zeros([h, w], dtype='int')
    for i in range(idx - 2):
        if eq_marks[i] == 0:  # 没有标注等价类，单独执行分支划分
            edge_merged[edge == i + 2] = label
            label += 1
            continue
        if eq_marks[i] not in branch_id_ori:
            branch_id_ori.append(eq_marks[i])
            branch_id_pre.append(label)
            edge_merged[edge == i + 2] = label
            label += 1
        else:
            edge_merged[edge == i + 2] = branch_id_pre[branch_id_ori.index(eq_marks[i])]
    return idx - 2 - eq_count, edge_merged


def fill_neighbor(edge, edge_ref):
    # 图像处理有效但逐点绘制无效的边缘点位置
    h, w = edge.shape
    candidate_pos = np.array(np.where((edge == 0) * (edge_ref != 0) == 1)).T
    candidate_strength = [edge_ref[pos[0], pos[1]] for pos in candidate_pos]
    candidate_ranked = np.argsort(candidate_strength)[::-1]
    for candidate_index in candidate_ranked:
        pos = candidate_pos[candidate_index, :]
        min_i, max_i, min_j, max_j = get_index(pos[0], pos[1], h, w, 1)
        edge_neighbor = edge[min_i:max_i + 1, min_j:max_j + 1]
        if branch_count_2path(edge_neighbor / 255, 8, return_branch_map=False) >= 2:
            neighbor_pos = np.where(edge_neighbor == 255)
            sum_line, sum_column = np.sum(neighbor_pos[0]), np.sum(neighbor_pos[1])
            mean_line = (sum_line + 1) / (len(neighbor_pos[0]) + 1)
            mean_column = (sum_column + 1) / (len(neighbor_pos[0]) + 1)
            edge[min_i + int(np.round(mean_line)), min_j + int(np.round(mean_column))] = 255
    return edge


def rotate_edge(edge, angle):
    """
    旋转边缘图指定角度
    :param edge: 边缘指示图，0/1组成
    :param angle: 逆时针旋转的角度
    :return: 旋转后的边缘图
    """
    angle_rad = angle / 180 * np.pi
    edge = edge[:, :, 0]
    edge_rotated = rotate(edge, angle)
    edge_pos = np.array(np.where(edge == 255)).T
    H_o, W_o = edge.shape
    H_p, W_p = edge_rotated.shape
    edge_pos_mid = edge_pos - np.array([[H_o / 2.0, W_o / 2.0]])
    edge_pos_rotated = [edge_pos_mid[:, 0] * np.cos(angle_rad) - edge_pos_mid[:, 1] * np.sin(angle_rad),
                        edge_pos_mid[:, 1] * np.cos(angle_rad) + edge_pos_mid[:, 0] * np.sin(angle_rad)]
    edge_pos_rotated = np.array(edge_pos_rotated).T
    edge_pos_rotated += np.array([[H_p / 2.0, W_p / 2.0]])
    pos_sorted = edge_pos_rotated[np.argsort(edge_pos_rotated[:, 0]), :]
    edge_pos_rotated = filter_neighbor(edge_pos_rotated)
    edge_rotated_point_wise = np.zeros_like(edge_rotated, dtype='uint8')
    for pos in edge_pos_rotated:
        edge_rotated_point_wise[
            int(np.clip(0, np.round(pos[0]), H_p - 1)), int(np.clip(0, np.round(pos[1]), W_p - 1))] = 255
    edge_final = fill_neighbor(edge_rotated_point_wise, edge_rotated)
    min_l, max_l, min_c, max_c = get_clip_range(edge.shape, angle)
    return edge_final[min_l:max_l, min_c:max_c, None].repeat(3, axis=2)


def scale_edge(edge, size):
    edge = edge[:, :, 0]
    h, w = edge.shape
    H, W = size
    ratio = H / h
    pos = np.array(np.where(edge == 255)).T
    pos_scaled = pos * ratio
    edge_scaled_interpolate = cv2.resize(edge, (W, H))
    edge_point_wise = np.zeros_like(edge_scaled_interpolate)
    for p in pos_scaled:
        line = int(np.clip(p[0], 0, H - 1))
        column = int(np.clip(p[1], 0, W - 1))
        edge_point_wise[line, column] = 255
    edge_filled = fill_neighbor(edge_point_wise, edge_scaled_interpolate)
    return edge_filled[:, :, None].repeat(3, axis=2)


def get_kernel(method, width, params):
    if method == 'unit':
        kernel = np.ones([width, width])
    elif method == 'tri':
        kernel = (np.ceil(width / 2) - np.abs(np.arange(1, width + 1) - (np.ceil(width / 2)))) / np.ceil(width / 2)
        kernel = conv(kernel[None, :], kernel[:, None])
    else:
        raise AttributeError('Undefined kernel %s' % method)
    return kernel


def image_blur(image: np.ndarray, method, width, params=None):
    """
    对图像进行糊化，起到对标注点的扩散作用
    :param image: C通道图像
    :param method: 糊化方法
    :param width: 糊化范围，即对 width*width 的范围进行扩散
    :param params: 糊化参数，应与 method 匹配
    """
    if len(image.shape) == 2:
        image = image[:, :, None]
    image = image.astype(np.float)
    kernel = get_kernel(method, width, params)
    image_blurred = [conv(image[:, :, c], kernel, mode='same')[:, :, None] for c in range(image.shape[2])]
    image_blurred = np.clip(np.concatenate(image_blurred, axis=2), 0, 255).astype('uint8')
    return image_blurred


def image_cover(image_base, mark_points, replace_channel=0):
    """
    在image_base中将mark_points非零的位置替换为对应值的红色像素
    """
    image = image_base.copy()
    pos = np.where(mark_points[:, :, 0] > 20)  # 较强元素值的位置
    for i in range(len(pos[0])):
        line, column = pos[0][i], pos[1][i]
        replace_color = [0, 0, 0]
        replace_color[replace_channel] = mark_points[line, column, 0]
        image[line, column, :] = replace_color
    return image


def process_image(image_path):
    """
    读取生成的关键点图像拼图，获取边缘和关键点标记
    :param image_path: 图像路径
    :return: 边缘，经非极大值抑制的关键点标记，交叉点、拐点和端点标记
    """
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) / 255
    H, W, _ = image.shape
    edge = image[:, int(W / 5):int(W / 5) * 2, 0]
    image_keypoint_nms = image[:, int(W / 5) * 3:int(W / 5) * 4, :]
    image_keypoint_bin = image[:, int(W / 5) * 4:, :]
    image_keypoint_bin: np.ndarray
    channel_R, channel_G, channel_B = image_keypoint_bin.transpose([2, 0, 1])
    flag_cross = channel_R * (channel_G == 0) * (channel_B == 0)
    flag_corner = (channel_R == 0) * channel_G * (channel_B == 0)
    flag_end = (channel_R == 0) * (channel_G == 0) * channel_B
    flag_keypoints = (flag_cross + flag_corner + flag_end) > 0
    pos_keypoint = np.where(flag_keypoints)
    return edge, image_keypoint_nms, flag_cross, flag_corner, flag_end, flag_keypoints, pos_keypoint


def generate_path(p1, p2):
    length = max(abs(p1[0] - p2[0]) + 1, abs(p1[1] - p2[1]) + 1)
    l_path = np.round(np.linspace(p1[0], p2[0], length)).astype('int')
    c_path = np.round(np.linspace(p1[1], p2[1], length)).astype('int')
    return l_path, c_path


def connected_straight(edge, p1, p2):
    # 连线上均为边缘点则认为有效连接
    l_path, c_path = generate_path(p1, p2)
    edge_path = edge[l_path, c_path]
    return all(edge_path > 0)


def connected_threshold(edge, p1, p2, threshold):
    # 连线上边缘强度的平均值高于阈值则认为有效连接
    l_path, c_path = generate_path(p1, p2)
    edge_path = edge[l_path, c_path]
    return np.mean(edge_path) >= threshold


def connected_weighted(edge, p1, p2):
    l_path, c_path = generate_path(p1, p2)
    edge_path = edge[l_path, c_path]
    return np.sum(edge_path)


def is_connected(edge, p1, p2, principle='straight'):
    if principle == 'straight':  # 直接连接，返回是否有连接的标志
        return connected_straight(edge, p1, p2)
    elif principle == 'uncovered':  # 后续需要检测覆盖，返回路径上的边缘强度和
        return connected_weighted(edge, p1, p2)
    elif principle in ['expand', 'degree']:  # 这里的边缘已经经过扩展，直接检测直线连接即可
        return connected_straight(edge, p1, p2)
    else:
        raise AttributeError("Undefined principle")


def get_valid_connections(connection, index):
    return np.sum(connection[index, :]) + np.sum(connection[:, index])


def get_connection(meta_coarse: SketchMeta, meta_shrink: SketchMeta, method):
    base, algorithm, principle = method.split('_')
    # 遍历每个关键点对
    if algorithm == "traversal":
        if base == 'coarse':
            edge, pos_keypoint = meta_coarse.edge, meta_coarse.pos_keypoint
        else:
            edge, pos_keypoint = meta_shrink.edge, meta_shrink.pos_keypoint
        N = len(pos_keypoint[0])
        if principle in ["expand", "degree"]:
            edge = conv(edge, np.array([[1/2, 1, 1/2]]) * np.array([[1/2, 1, 1/2]]).transpose(), mode='same')
        max_connections = meta_shrink.flag_cross * 4 + meta_shrink.flag_corner * 2 + meta_shrink.flag_end * 1
        connections = np.zeros([N, N])
        for i in range(N):
            p1 = [pos_keypoint[0][i], pos_keypoint[1][i]]
            max_connection_p1 = max_connections[p1[0], p1[1]]
            for j in range(i + 1, N):
                p2 = [pos_keypoint[0][j], pos_keypoint[1][j]]
                max_connection_p2 = max_connections[p2[0], p2[1]]
                if get_valid_connections(connections, i) >= max_connection_p1 \
                        or get_valid_connections(connections, j) >= max_connection_p2:
                    continue
                # 检测是否有连接
                connections[i, j] = is_connected(edge, p1, p2, principle)
        return pos_keypoint, connections


def get_performance(edge, sketch):
    # 边缘覆盖率
    r = 4
    sketch_mask = conv(sketch, np.ones([2 * r + 1, 2 * r + 1]), mode='same') > 0
    edge_ratio = np.sum(edge * sketch_mask) / (np.sum(edge) + 1e-30)
    # 素描覆盖率
    edge_mask = conv(edge, np.ones([2 * r + 1, 2 * r + 1]), mode='same') > 0
    sketch_ratio = np.sum(edge_mask * sketch) / (np.sum(sketch) + 1e-30)
    # 平均素描距离
    pos_sketch = np.array(np.where(sketch > 0)).transpose()  # 素描点的坐标
    if pos_sketch.shape[0] == 0:  # 没有素描点时添加[0,0]以避免计算错误
        pos_sketch = np.array([[0, 0]])
    pos_edge = np.array(np.where(edge > 0)).transpose()  # 边缘点的坐标
    distances = 0
    for pos in pos_edge:
        distance_min = np.min(np.sqrt(np.sum((pos_sketch - pos) ** 2, axis=1)))
        distances += distance_min * edge[pos[0], pos[1]]  # 距离加权与归一化
    distance_mean = distances / pos_edge.shape[0]
    return {'edge_ratio': float(edge_ratio),
            'sketch_ratio': float(sketch_ratio),
            'average_distance': float(distance_mean)}


def write_RGB(path, image):
    """
    写图像
    :param path: 写入路径
    :param image: RGB三通道、RGBA四通道或单通道
    :return:
    """
    shape = image.shape
    assert len(shape) >= 2
    if len(shape) == 2:
        H, W = shape
    else:
        H, W, C = shape
        if C >= 3:
            image[:, :, :3] = image[:, :, :3][:, :, ::-1]  # convert RGB to BGR
    cv2.imwrite(path, image)


def merge_performance(performances: dict):
    keys = list(performances.keys())
    indicates = performances[keys[0]].keys()
    for indicate in indicates:
        performances.update({'%s_mean' % indicate: float(np.mean([performances[key][indicate] for key in keys]))})
    return performances


def print_performance_item(file_name, time_cost, performance):
    print()
    print('image: %s | '
          'time: %2.2f | '
          'edge_ratio: %.2f | '
          'sketch_ratio: %.2f | '
          'distance: %.2f'
          % (file_name,
             time_cost,
             performance['edge_ratio'],
             performance['sketch_ratio'],
             performance['average_distance']
             ))


def print_performance_summary(performances):
    print('average time: %2.2f | '
          'edge_ratio: %.2f | '
          'sketch_ratio: %.2f | '
          'average distance: %.2f'
          % (performances['time_mean'],
             performances['edge_ratio_mean'],
             performances['sketch_ratio_mean'],
             performances['average_distance_mean']
             ))


def deal_with_parameters(base, algorithm, principle):
    assert base in ['coarse', 'shrink', 'mix']
    assert algorithm in ['traversal']
    assert principle in ['straight', 'uncovered', 'expand', 'degree']
    method = "%s_%s_%s" % (base, algorithm, principle)
    if principle == 'uncovered':
        raise AttributeError('Not realized yet!')
    print(method)
    return method


def get_up_index(N):
    """
    获得 NxN 方阵上三角阵（不含对角线）的序号，如 1,2,3,...,9,12,13,...,19,23,...,29,...
    """
    indexes = np.arange(0, N*N).reshape([N, N])
    indexes_up = indexes[np.triu_indices(N, k=1, m=None)]
    return indexes_up


def get_undistort_params(params, h, w):
    rx, ry, k1, k2, k3, p1, p2 = params
    # 相机焦距
    fx = 685.646752
    fy = 676.658033
    # 相机中心位置
    cx = int(w*rx)
    cy = int(h*ry)
    # 相机坐标系到像素坐标系的转换矩阵
    k = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    # 畸变系数
    d = np.array([
        k1, k2, p1, p2, k3
    ])
    mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), cv2.CV_32FC1)
    return mapx, mapy


def find_inverse_map(mapx, mapy, positions, h, w, return_Tensor=False):
    maps = torch.cat([torch.Tensor(mapy[:, :, None]), torch.Tensor(mapx[:, :, None])], 2).cuda()
    if positions.__class__ != torch.Tensor:
        positions_target = torch.Tensor(positions).cuda()
    else:
        positions_target = positions.cuda()
    distances = torch.sum(torch.abs(maps[:, :, None, :] - positions_target[None, None, :, :]), dim=3)
    index = torch.argmin(distances.reshape([-1, len(positions)]), dim=0)
    line = index // w
    column = index % w
    valid = ((distances[line, column, range(len(positions))] < 2) + (line > 0) * (line < h - 1) * (column > 0) * (column < w - 1)) > 0
    if return_Tensor:
        positions_distort = torch.cat([line[:, None], column[:, None]], dim=1).to(torch.int)
    else:
        positions_distort = np.array([line.cpu().numpy(), column.cpu().numpy()]).transpose().astype(np.int)
    return positions_distort, valid


def undistort(frame, positions, params, return_Tensor=False):
    params = list(params)
    h, w = frame.shape[:2]
    # 失真参数，k表示径向畸变， p表示切向畸变
    mapx, mapy = get_undistort_params(params, h, w)
    positions_distort, valid = find_inverse_map(mapx, mapy, positions, h, w, return_Tensor)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR), positions_distort, valid


def distance_sketch(coordinates, pos_keypoints, p1_index, p2_index):
    """
计算N个点到M对连线的距离
    :param coordinates: 待考察的N个点坐标，Nx2
    :param pos_keypoints: K*2，K 个关键点的坐标
    :param p1_index: M，M 组连线端点之一的序号
    :param p2_index: M，M 组连线端点之二序号
    :return: 距离，NxM
    """
    if coordinates.__class__ == np.ndarray:
        d_outer_all = np.sqrt(np.sum(np.square(coordinates[:, None, :] - pos_keypoints[None, :, :]), axis=2))  # NxK
        d1 = d_outer_all[:, p1_index]  # NxM
        d2 = d_outer_all[:, p2_index]  # NxM
        d_inner_all = np.sqrt(np.sum(np.square(np.triu((pos_keypoints[None, :, :] - pos_keypoints[:, None, :]).transpose(1, 2, 0), 1)), axis=2))  # KxK
        d = d_inner_all[p1_index, p2_index]  # M
        sketch_distance = (d1 + d2) / (d[None, :] + 1e-30)
    else:
        assert coordinates.__class__ == torch.Tensor
        d_outer_all = torch.sqrt(torch.sum(torch.square(coordinates[:, None, :] - pos_keypoints[None, :, :]), dim=2))  # NxK
        d_inner_all = torch.sqrt(torch.sum(torch.square(torch.triu((pos_keypoints[None, :, :] - pos_keypoints[:, None, :]).permute(2, 0, 1), 1)), dim=0))  # KxK
        d = d_inner_all[p1_index, p2_index]  # M
        sketch_distance = (d_outer_all[:, p1_index] + d_outer_all[:, p2_index]) / \
                          (d[None, :] + torch.Tensor([1e-30]).to(coordinates.device))
    return sketch_distance


def sketch_score(coordinates, pos_keypoints, p1_index, p2_index, lmbda=1.0):
    """
计算所给坐标到连线的分数
    :param coordinates: N*2，待考察的 N 个坐标点，两个维度分别表示行数和列数
    :param pos_keypoints: K*2，K 个关键点的坐标
    :param p1_index: M，M 组连线端点之一的序号
    :param p2_index: M，M 组连线端点之二序号
    :param lmbda: 系数，控制分数的陡峭程度
    :return: N*M，N个坐标到 M 组连线的分数
    """
    sketch_distance = distance_sketch(coordinates, pos_keypoints, p1_index, p2_index)  # NxM
    if sketch_distance.__class__ == np.ndarray:
        scores = np.exp(-(sketch_distance-1)/lmbda)
    else:
        assert sketch_distance.__class__ == torch.Tensor
        scores = torch.exp(-(sketch_distance-1)/torch.Tensor([lmbda]).to(sketch_distance.device))
    return scores  # NxM


def generate_sketch(coordinates: torch.Tensor, connections: torch.Tensor, edge, size, lmbda=0.01):
    """
根据所给关键点坐标和连接关系矩阵绘制连续性素描图，每一条素描线表现为杵状
    :param coordinates: 关键点坐标
    :param connections: 关键点间两两的连接关系
    :param edge: HxW，图像实际边缘，起到调整连线权重的作用
    :param size: 绘制的图像尺寸
    :param lmbda: 连线描绘参数，决定连线的粗细程度
    :return: HxW 素描图
    """
    assert len(coordinates.shape) == 2
    assert len(connections.shape) == 2
    assert coordinates.shape[0] == connections.shape[0] == connections.shape[1]
    assert coordinates.shape[1] == 2
    H, W = size
    N = coordinates.shape[0]
    connections = torch.triu(connections, 1)
    indexes_sorted = torch.argsort(-connections.flatten())
    line_sorted = indexes_sorted // N
    column_sorted = indexes_sorted % N
    # p1_indexes, p2_indexes = torch.where(connections > 0.01)  # 按阈值选择
    p1_indexes, p2_indexes = line_sorted[:200], column_sorted[:200]  # 按数量选择
    p1 = coordinates[p1_indexes, :]
    p2 = coordinates[p2_indexes, :]
    weights = torch.sqrt(edge[p1[:, 0], p1[:, 1]] * edge[p2[:, 0], p2[:, 1]])
    conn_strength = connections[p1_indexes, p2_indexes]  # N
    indexes_pixels = torch.arange(H * W).reshape([H, W]).to(coordinates.device)
    ls = indexes_pixels // W
    cs = indexes_pixels % W
    coordinates_pixels = torch.stack([ls, cs]).reshape([2, -1]).transpose(0, 1)
    scores = sketch_score(coordinates_pixels, coordinates, p1_indexes, p2_indexes, lmbda=lmbda)  # PxN，P个像素点到N对连线的分数
    sketch_scores = scores * conn_strength[None] * weights[None]  # PxN，加权后的分数
    sketch_strength = sketch_scores.max(dim=1)[0].reshape([H, W])
    return sketch_strength/torch.max(sketch_strength)  # 归一化到 0~1


def get_orientations(edge, kernel_size=5, splits=16, return_response=False):
    """
基于 Pytorch 实现对边缘方向的检测，即初始化不同方向的滤波器，根据滤波器响应来决定边缘的方向
    :param edge: 输入边缘图像，归一化到 0~1 之间，边缘点为非 0 点. HxW
    :param kernel_size: 滤波器核大小，为奇数值
    :param splits: 方向检测的粒度，即将 180 度几等分
    :return: 每个边缘点的方向，0~180°
    """
    if edge.__class__ != torch.Tensor:
        edge_tensor = torch.Tensor(edge)[None, None]
    else:
        edge_tensor = edge[None, None]
    parent_size = int(np.ceil(kernel_size*np.sqrt(2))) // 2 * 2 + 1  # 母核大小，防止旋转后边界处衰减
    padding_size = int(kernel_size / 2)  # padding 大小
    initial_kernel = torch.zeros([1, 1, parent_size, parent_size])
    initial_kernel[0, 0, int((parent_size-1)/2), :] = 1  # 角度为 0 的核，作为初始核
    angles = np.linspace(0, 180, splits, endpoint=False)  # 候选方向的角度
    kernels_orientation = torch.zeros([splits, 1, parent_size, parent_size])
    for s in range(splits):  # 对初始核按照不同角度旋转，获得不同方向的滤波器核
        kernels_orientation[s, 0] = TF.rotate(initial_kernel, angles[s], interpolation=TF.InterpolationMode.BILINEAR)
    kernels_orientation = kernels_orientation[:, :,
                          int((parent_size - kernel_size)/2):-int((parent_size - kernel_size)/2),
                          int((parent_size - kernel_size)/2):-int((parent_size - kernel_size)/2)]
    edge_response = F.conv2d(edge_tensor, kernels_orientation.to(edge_tensor.device), padding=[padding_size, padding_size])
    channel_activate = torch.argmax(edge_response, dim=1)[0]
    edge_angles = angles[channel_activate.cpu().numpy()] * (edge_tensor[0,0] > 0).to(torch.float).cpu().numpy()
    if return_response:
        return edge_angles, edge_response
    return edge_angles


def get_distance_from_point_to_line(point, line_point1, line_point2):
    # 计算直线的三个参数
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]
    # 根据点到直线的距离公式计算距离
    if point.__class__ == torch.Tensor:
        distance = torch.abs(A * point[0] + B * point[1] + C) / (torch.sqrt(A ** 2 + B ** 2) + 1e-6)
    else:
        distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2)+1e-6)
    return distance


def dis_point_to_seg_line(p, a, b):
    """
点到线段的距离
    """
    a, b, p = np.array(a), np.array(b), np.array(p)	 # trans to np.array
    d = np.divide(b - a, np.linalg.norm(b - a))	 # normalized tangent vector
    s = np.dot(a - p, d)  # signed parallel distance components
    t = np.dot(p - b, d)
    h = np.maximum.reduce([s, t, 0])  # clamped parallel distance
    c = np.cross(p - a, d)  # perpendicular distance component
    return np.hypot(h, np.linalg.norm(c))


def svg2png(svg_path, save_png_path, target_width=1):
    pic = svg2rlg(svg_path)
    for i in range(len(pic.contents[0].contents[0].contents[0].contents)):
        pic.contents[0].contents[0].contents[0].contents[i].contents[0].strokeWidth = target_width  # 将线条宽度设置为指定数值
    renderPM.drawToFile(pic, save_png_path)


class PerformanceRecorder:
    def __init__(self):
        self.metric = {}
        self.keys = []
        self.count = 0

    def update(self, metric_dict):
        if not self.metric:
            self.metric = metric_dict
            self.keys = metric_dict.keys()
            self.count += 1
        else:
            self.metric = {key: self.metric[key] * self.count + metric_dict[key] for key in self.keys}
            self.count += 1
            self.metric = {key: self.metric[key] / self.count for key in self.keys}


def assign_classes(args):
    if args.classes == '':
        return None  # 不指定类别，则使用所有类别
    if args.classes.lower() == 'random':  # 随机选取类别，则由指定随机种子生成
        classes_all = classes_HumanSketch if args.dataset.lower() == 'humansketch' \
            else classes_QuickDraw if args.dataset.lower() == 'quickdraw' \
            else classes_ImageNet if 'imagenet' in args.dataset.lower() \
            else classes_Caltech if 'caltech' in args.dataset.lower() \
            else classes_Cifar if 'cifar' in args.dataset.lower() \
            else classes_Fasion
        classes_count_all = len(classes_all)
        if args.class_count > classes_count_all:
            args.class_count = classes_count_all
            print('Warning: Totally %d classes, and required %d.' % (classes_count_all, args.class_count))
        np.random.seed(513)
        sample_indexes = np.random.choice(classes_count_all, args.class_count, replace=False)
        classes_sampled = list(np.array(classes_all)[sample_indexes])
        classes_sampled.sort()
        return classes_sampled
    # 否则采用指定的类别
    classes = args.classes.split(',')
    classes = [c for c in classes if c]
    return classes  # 指定类别，则返回所指定的类别


class SketchRecorder:
    """
记录素描图检测过程中各个环节花费的时间
    """
    def __init__(self):
        self.recorder_keypoint = AverageMeter()
        self.recorder_connection = AverageMeter()
        self.recorder_select_tri = AverageMeter()
        self.recorder_select_bypass = AverageMeter()
        self.recorder_makeup = AverageMeter()
        self.recorder_plot = AverageMeter()

    def update_keypoint(self, t, n=1):
        self.recorder_keypoint.update(t, n)

    def update_connection(self, t, n=1):
        self.recorder_connection.update(t, n)

    def update_select_tri(self, t, n=1):
        self.recorder_select_tri.update(t, n)

    def update_select_bypass(self, t, n=1):
        self.recorder_select_bypass.update(t, n)

    def update_makeup(self, t, n=1):
        self.recorder_makeup.update(t, n)

    def update_plot(self, t, n=1):
        self.recorder_plot.update(t, n)


def get_reduction_sketch_distance(positions, distances_ori, p1, p2):
    """
计算素描距离的减少量，以及得到减少的序号和当前素描距离
    :param positions: 考察点的坐标，Nx2
    :param distances_ori: 各个考察点的原始素描距离，N
    :param p1: 考察素描点1
    :param p2: 考察素描点2
    :return: 1. 素描距离的减少量总和；
             2. 素描距离得到减少的考察点序号
             3. 2.中考察点当前的素描距离
    """
    distances_now = np.array([dis_point_to_seg_line(p, p1, p2) for p in positions])
    reduced_indexes = np.where(distances_now < distances_ori)[0]
    distances_reduced_now = distances_now[reduced_indexes]
    reduced_distances = np.sum(distances_ori[reduced_indexes] - distances_reduced_now)
    return reduced_distances, reduced_indexes, distances_reduced_now


def get_upper_sketch_distance(p1, p2):
    L = np.sqrt(np.sum(np.square(p1 - p2)))
    N = max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1])) + 1
    return (N-1)*L/4


def get_nearest(target_positions, candidate_positions):
    """
从候选点中获得离目标点最近的点及索引
    :param target_positions: 目标点，N1x2
    :param candidate_positions: 候选点，N2x2
    :return: 最近候选点的索引和坐标，维度分别为N1,N1x2
    """
    distances = np.sum(np.square(target_positions[:, None] - candidate_positions[None]), axis=2)
    nearest_indexes = np.argmin(distances, axis=1)
    nearest_positions = candidate_positions[nearest_indexes]
    return nearest_indexes, nearest_positions


def merge_list(array_of_list):
    a = []
    for b in array_of_list:
        a.extend(b)
    return a


def get_redundant_groups(redundant_plot_pairs):
    """
将各个冗余点所属的素描点对归纳到一组
    """
    redundant_groups = []
    for pair in redundant_plot_pairs:
        if len(pair[0]) == 1:
            continue
        pairs_rearranged = [(i1, i2) for (i1, i2) in zip(pair[0], pair[1])]
        redundant_groups.append(pairs_rearranged)
    return redundant_groups


def get_overlap(redundant_pairs, redundant_groups):
    """
计算各个素描点对的覆盖度，即所绘制点的重复数目
    """
    redundant_overlap_count = np.zeros((len(redundant_pairs), len(redundant_pairs)))
    for redundant_group in redundant_groups:
        for i in range(len(redundant_group)):
            for j in range(i+1, len(redundant_group)):
                index_i = redundant_pairs.index('%d,%d' % (redundant_group[i][0], redundant_group[i][1]))
                index_j = redundant_pairs.index('%d,%d' % (redundant_group[j][0], redundant_group[j][1]))
                redundant_overlap_count[index_i, index_j] += 1
    return redundant_overlap_count


def revise_sketch(edge, sketch, sketch_graph, sketch_pairs, nms_agent):
    a = 1
    positions, connections = sketch_graph['positions'], sketch_graph['connections']
    # orientations = get_orientations(edge, splits=32) / 180 * np.pi  # 检测粒度为 180/32 度即约6度
    # N = positions.shape[0]
    """part1: 冗余连接关系去除"""
    # step1: 消除错误冗余
    # 获取有效点坐标
    points_edge = np.array(np.where(edge)).transpose()
    points_draw = np.array(np.where(sketch)).transpose()
    # 计算绘制点到边缘点的距离
    distances_edge_draw_pairs = np.sqrt(np.sum(np.square(np.abs(points_edge[:, None, :] - points_draw[None, :, :])), axis=2))
    distances_draw = np.min(distances_edge_draw_pairs, axis=0)  # 各个绘制点到边缘的最小距离
    # 对绘制点到边缘的距离可视化
    distances_draw_vision = np.zeros(edge.shape)
    distances_draw_vision[points_draw[:, 0], points_draw[:, 1]] = distances_draw
    # 根据阈值设置素描点对的权重
    threshold_draw = 2  # 绘制点到边缘的距离阈值
    redundant_pairs_count = np.zeros_like(connections)
    pairs_count = np.zeros_like(connections)
    redundant_indexes = np.where(distances_draw > threshold_draw)[0]  # 产生冗余的绘制点的序号
    redundant_points = points_draw[redundant_indexes]  # 产生冗余的绘制点的坐标
    redundant_pairs = sketch_pairs[redundant_points[:, 0], redundant_points[:, 1]]  # 产生冗余的素描点序号
    for pair in redundant_pairs:
        redundant_pairs_count[pair[0], pair[1]] += 1
    for point in points_draw:
        if sketch_pairs[point[0], point[1], 0] != -1:  # 孤立关键点也是绘制点，但连接点序号是(-1,-1)
            pairs_count[sketch_pairs[point[0], point[1], 0], sketch_pairs[point[0], point[1], 1]] += 1
    points_redundant_ratio = redundant_pairs_count / (pairs_count + 1e-5)  # 冗余点占绘制点的比例
    # 将冗余率高于阈值的连接关系置为0
    redundant_pairs = np.array(np.where(points_redundant_ratio > 0.3)).transpose()  # 冗余率超过阈值的连接点对
    connections[redundant_pairs[:, 0], redundant_pairs[:, 1]] = 0
    connections[redundant_pairs[:, 1], redundant_pairs[:, 0]] = 0
    # 重新绘制
    sketch, plot_pairs, plot_times = plot_sketch(positions, (800, 800), connections, threshold_connection=0.3)
    # step2: 消除重复冗余
    redundant_flags = (plot_times > 1) * (sketch[:, :, 0] != 255)  # 冗余点不计素描点
    if redundant_flags.any():
        redundant_plot_times = plot_times[redundant_flags]
        redundant_plot_pairs = plot_pairs[redundant_flags]
        redundant_plot_pairs_line = merge_list(redundant_plot_pairs[:, 0])
        redundant_plot_pairs_column = merge_list(redundant_plot_pairs[:, 1])
        redundant_plot_pairs_merged = np.array([redundant_plot_pairs_line, redundant_plot_pairs_column]).transpose()
        redundant_pairs_counter = Counter(['%d,%d' % (a[0], a[1]) for a in redundant_plot_pairs_merged])
        redundant_pairs = list(redundant_pairs_counter.keys())  # 冗余点对
        redundant_pairs.sort()
        redundant_pairs_count = np.array([redundant_pairs_counter[key] for key in redundant_pairs])  # 冗余点对的冗余数目
        redundant_pairs_length = np.zeros((len(redundant_pairs_count,)))
        for idx, redundant_pair in enumerate(redundant_pairs):
            i1_str, i2_str = redundant_pair.split(',')
            p1, p2 = positions[[int(i1_str), int(i2_str)]]
            redundant_pairs_length[idx] = max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])) - 1
        redundant_pairs_ratio = redundant_pairs_count / redundant_pairs_length
        redundant_groups = get_redundant_groups(redundant_plot_pairs)  # 各个冗余点所连接的冗余点对
        # redundant_overlap_count = get_overlap(redundant_pairs, redundant_groups)  # 各个素描点对的重叠数量，在取消一个点对时对另外的点对产生影响
        ratio_max_idx = np.argmax(redundant_pairs_ratio)
        threshold_redundant = 0.3
        # t_s = time.time()
        while redundant_pairs_ratio[ratio_max_idx] > threshold_redundant:
            i1, i2 = redundant_pairs[ratio_max_idx].split(',')  # 该冗余度对应的素描点的序号
            i1, i2 = int(i1), int(i2)
            # print('removed a pair: (%d,%d)' % (i1, i2))
            connections[i1, i2] = 0
            connections[i2, i1] = 0  # 将冗余度最高的点对取消连接关系
            redundant_pairs_count[ratio_max_idx] = 0  # 该点对的冗余数目置为0
            # 考察所有冗余点，若该连接关系经过一冗余点，则将该点对从该冗余点的所属点对中移除，该点的绘制次数减一；
            # 若冗余点绘制次数减为1，则对其所属的点对的冗余数减1。
            # pass_count = 0  # 点对经过冗余点的数目有上限，用于提前终止（不显著）
            for idx, redundant_group in enumerate(redundant_groups):
                # if pass_count == redundant_pairs_length[ratio_max_idx]:
                #     break
                if redundant_plot_times[idx] == 1:  # 已经不是冗余点，不进行后续处理
                    continue
                if (i1, i2) in redundant_group:  # 点对经过这个冗余点
                    # pass_count += 1
                    redundant_group.remove((i1, i2))  # 将冗余点的所属点对中移除该点对
                    redundant_plot_times[idx] -= 1
                    if redundant_plot_times[idx] == 1:  # 该点绘制次数变为1，则不是冗余点。将其所属点对的冗余次数减一
                        redundant_pairs_count[redundant_pairs.index('%d,%d' % (redundant_group[0][0], redundant_group[0][1]))] -= 1
            redundant_pairs_ratio = redundant_pairs_count / redundant_pairs_length  # 重新计算冗余比例
            ratio_max_idx = np.argmax(redundant_pairs_ratio)  # 冗余比例最大的点对
        # print(time.time() - t_s)
        plot_times[redundant_flags] = redundant_plot_times
        sketch, plot_pairs, _ = plot_sketch(positions, (800, 800), connections, threshold_connection=0.3)
    # Step3: 消除并排冗余
    sketch_tensor = torch.Tensor(sketch[:, :, 0])[None, None]
    sketch_nms = nms_agent(sketch_tensor)
    suppression_flags = (sketch_tensor > 0) * (sketch_nms == 0) * (plot_pairs[:, :, 0] != -1)  # 被抑制的标志
    # 统计被抑制的点对
    if suppression_flags.any():
        suppression_pairs_0 = plot_pairs[suppression_flags[0, 0].to(torch.bool)]
        suppression_pairs_line = merge_list(suppression_pairs_0[:, 0])
        suppression_pairs_column = merge_list(suppression_pairs_0[:, 1])
        suppression_pairs_merged = np.array([suppression_pairs_line, suppression_pairs_column]).transpose()
        suppression_pairs_counter = Counter(['%d,%d' % (a[0], a[1]) for a in suppression_pairs_merged])
        suppression_pairs = list(suppression_pairs_counter.keys())  # 冗余点对
        suppression_pairs.sort()
        suppression_pairs_count = np.array([suppression_pairs_counter[key] for key in suppression_pairs])  # 冗余点对的冗余数目
        suppression_pairs_length = np.zeros((len(suppression_pairs_count,)))
        for idx, suppression_pair in enumerate(suppression_pairs):
            i1_str, i2_str = suppression_pair.split(',')
            p1, p2 = positions[[int(i1_str), int(i2_str)]]
            suppression_pairs_length[idx] = max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])) + 1
        suppression_pairs_ratio = suppression_pairs_count / (suppression_pairs_length + 1e-5)
        ratio_max_idx = np.argmax(suppression_pairs_ratio)
        threshold_suppression = 0.4
        while suppression_pairs_ratio[ratio_max_idx] > threshold_suppression:
            i1, i2 = suppression_pairs[ratio_max_idx].split(',')  # 该冗余度对应的素描点的序号
            i1, i2 = int(i1), int(i2)
            # print('removed a pair: (%d,%d) with ratio: %.3f' % (i1, i2, suppression_pairs_ratio[ratio_max_idx]))
            # print('positions: ', positions[[i1, i2]])
            connections[i1, i2] = 0
            connections[i2, i1] = 0  # 将冗余度最高的点对取消连接关系
            suppression_pairs_count[ratio_max_idx] = 0  # 该点对的冗余数目置为0
            suppression_pairs_ratio = suppression_pairs_count / (suppression_pairs_length + 1e-5)  # 重新计算冗余比例
            ratio_max_idx = np.argmax(suppression_pairs_ratio)  # 冗余比例最大的点对
    """part2: 缺失连接关系补充"""
    sketch = plot_sketch(positions, (800, 800), connections, threshold_connection=0.3)[0]
    # 计算边缘点到绘制点的距离
    points_draw = np.array(np.where(sketch.astype('int64').sum(axis=2))).transpose()
    distances_edge_draw_pairs = np.sqrt(np.sum(np.square(np.abs(points_edge[:, None, :] - points_draw[None, :, :])), axis=2))
    distances_edge = np.min(distances_edge_draw_pairs, axis=1)  # 各个边缘点到绘制点的最小距离
    # 对边缘点到绘制点的距离可视化
    distances_edge_vision = np.zeros(edge.shape)
    distances_edge_vision[points_edge[:, 0], points_edge[:, 1]] = distances_edge
    # 方案一：根据距离遍历求解（舍弃）
    # # 设定阈值并标注缺失点
    # threshold_lack, threshold_makeup, threshold_append = 2, 4, 10
    # threshold_ratio = 0.8  # 素描增益阈值
    # threshold_orientation = np.pi/18  # 角度阈值，不考虑与当前边缘点边缘方向相差超过10度的点对连线
    # lack_flag = distances_edge >= threshold_lack  # 各个边缘点是否缺失的标记
    # lack_indexes = np.where(lack_flag)[0]  # 缺失点的序号
    # lack_indexes_ranked_relative = np.argsort(-distances_edge[lack_indexes])  # 缺失点按照距离由大到小的排序（相对）
    # # 按距离由高到低开始考察
    # for lack_index_relative in lack_indexes_ranked_relative:
    #     lack_index = lack_indexes[lack_index_relative]  # 考察缺失点的序号
    #     if not lack_flag[lack_index]:  # 当前考察点受其他考察点的影响而取消缺失标志
    #         continue
    #     lack_position = points_edge[lack_index]  # 缺失点的坐标
    #     distances2sketch_points = np.sum(np.square(lack_position - positions), axis=1)  # 缺失点到各个素描点的距离
    #     candidate_sketch_indexes = np.argsort(distances2sketch_points)  # 按照距离由近到远对素描点排序
    #     # 寻找素描点对
    #     past_sketch_indexes = [candidate_sketch_indexes[0]]  # 历史点
    #     success = False
    #     candidate_pair, candidate_reduction_ratio, candidate_reduced_lack_indexes_relative, \
    #         candidate_lack_sketch_distance, candidate_distance2sketch_pairs \
    #         = [], 0, [], 0, 0
    #     for sketch_index in candidate_sketch_indexes[1:]:  # 依次考察素描点
    #         for past_index in past_sketch_indexes:
    #             orientation_line = np.arctan((positions[sketch_index, 1] - positions[past_index, 1])/(positions[sketch_index, 0] - positions[past_index, 0] + 1e-5)) + np.pi/2  # (0~pi)
    #             orientation_error = (orientation_line - orientations[lack_position[0], lack_position[1]]) % np.pi  # 角度之差，归至0~pi
    #             orientation_error = np.pi - orientation_error if orientation_error > np.pi/2 else orientation_error  # 夹角，归至0~pi/2
    #             if orientation_error > threshold_orientation:
    #                 continue
    #             distance2sketch_pairs = dis_point_to_seg_line(lack_position, positions[sketch_index], positions[past_index])
    #             promising = False
    #             if distance2sketch_pairs < threshold_makeup:  # 距离小于阈值
    #                 promising = True
    #                 reduction_sketch_distance, reduced_lack_indexes_relative, lack_sketch_distance = get_reduction_sketch_distance(points_edge[lack_indexes], distances_edge[lack_indexes], positions[sketch_index], positions[past_index])  # 点对带来的素描距离减少量
    #                 sketch_distance_upper = get_upper_sketch_distance(positions[sketch_index], positions[past_index])  # 点对的理论素描距离减少量
    #                 reduction_ratio = reduction_sketch_distance / (sketch_distance_upper+1e-5)  # 素描距离增益
    #                 if reduction_ratio > threshold_ratio:  # 素描增益大于阈值
    #                     success = True
    #                     # 连接关系置为有效
    #                     connections[sketch_index, past_index] = 1
    #                     connections[past_index, sketch_index] = 1
    #                     # 标记其他缺失点：将缺失点中当前素描距离小于考察点素描距离者取消缺失标记（设置一定的容错）
    #                     lack_flag[lack_indexes[reduced_lack_indexes_relative[lack_sketch_distance <= distance2sketch_pairs*1.5+1]]] = 0
    #             elif distance2sketch_pairs < threshold_append:  # 距离大于小阈值但小于大阈值，仍作为后续备选素描点
    #                 promising = True
    #             if promising:
    #                 reduction_sketch_distance, reduced_lack_indexes_relative, lack_sketch_distance = get_reduction_sketch_distance(
    #                     points_edge[lack_indexes], distances_edge[lack_indexes], positions[sketch_index],
    #                     positions[past_index])  # 点对带来的素描距离减少量
    #                 sketch_distance_upper = get_upper_sketch_distance(positions[sketch_index],
    #                                                                   positions[past_index])  # 点对的理论素描距离减少量
    #                 reduction_ratio = reduction_sketch_distance / sketch_distance_upper  # 素描距离增益
    #                 if reduction_ratio > candidate_reduction_ratio:
    #                     candidate_reduction_ratio = reduction_ratio
    #                     candidate_pair = [sketch_index, past_index]  # 记录候选素描点对
    #                     candidate_reduced_lack_indexes_relative = reduced_lack_indexes_relative  # 记录此时得到优化的缺失点的相对序号
    #                     candidate_lack_sketch_distance = lack_sketch_distance  # 记录得到优化的缺失点此时的素描距离
    #                     candidate_distance2sketch_pairs = distance2sketch_pairs
    #             if success:  # 用当前历史点找到了点对，不用再考虑其他历史点
    #                 break
    #         if success:  # 对于当前素描点找到了点对，不用再考虑后面的素描点
    #             break
    #         else:  # 当前素描点没有找到点对，则加入历史点，考察下一个素描点
    #             past_sketch_indexes.append(sketch_index)
    #     if not success:  # 考察了前20个素描点仍未匹配成功，则添加一个素描点，且与其对应的候选点连接
    #         if candidate_reduced_lack_indexes_relative == []:
    #             continue
    #         # 添加一个素描点
    #         positions = np.vstack([positions, lack_position[None]]).copy()
    #         connections = np.vstack([np.hstack([connections, np.zeros((N, 1))]), np.zeros((1, N+1))])
    #         connections[candidate_pair, N] = 1
    #         connections[N, candidate_pair] = 1
    #         N += 1
    #         # 将相关缺失点置为不缺失
    #         lack_flag[
    #             lack_indexes[candidate_reduced_lack_indexes_relative[candidate_lack_sketch_distance <= candidate_distance2sketch_pairs * 1.5]]] = 0
    threshold_lack = 2
    # 缺失补充方案二：根据缺失点拟合求解
    lack_flags = distances_edge_vision >= threshold_lack  # 各个点是否缺失的标记
    branch_count, branch_labels = branch_count_2path(lack_flags, neighbor_range=8, return_branch_map=True)
    for branch_idx in range(branch_count):
        lack_positions = np.array(np.where(branch_labels == branch_idx+1)).transpose()
        distances = np.sum(np.square(lack_positions[None] - lack_positions[:, None]), axis=2)
        end_idxes = np.where(distances == np.max(distances))[0]  # 距离最远的两个点的序号
        nearest_sketch_indexes, _ = get_nearest(lack_positions[end_idxes], positions)  # 获取离两个缺失点最近的素描点
        if len(nearest_sketch_indexes) == 1:
            nearest_sketch_indexes = np.concatenate([nearest_sketch_indexes, nearest_sketch_indexes])
        if nearest_sketch_indexes[0] != nearest_sketch_indexes[1]:
            connections[nearest_sketch_indexes[0], nearest_sketch_indexes[1]] = 1
            connections[nearest_sketch_indexes[1], nearest_sketch_indexes[0]] = 1
    sketch = plot_sketch(positions,
                         edge.shape,
                         connections,
                         threshold_connection=0.3)[0]
    sketch_graph = {'positions': positions, 'connections': connections}
    return sketch, sketch_graph


def save_graph(graph, save_path):
    threshold = 0.3
    graph['positions'] = [[int(a[0]), int(a[1])] for a in graph['positions']]
    graph['connections'] = [[int(i > threshold) for i in a] for a in graph['connections']]
    with open(save_path, 'w') as f:
        json.dump(graph, f)


def group_neighbored(neighbored_flag):
    neighbored_numbers = neighbored_flag.sum(axis=1)
    candidate_indexes = np.where(neighbored_numbers)[0]  # 与其他点存在相邻关系的点序号，即考察点
    neighbored_groups = []
    end_indexes_relative = np.where(neighbored_numbers[candidate_indexes] == 1)[0]  # 只与一个点相邻，即一条折线的端点
    marked = np.zeros_like(neighbored_numbers, dtype=bool)  # 每个考察点被标记过的标志
    for end_index_relative in end_indexes_relative:
        if marked[candidate_indexes[end_index_relative]]:
            continue
        neighbored_branch = [candidate_indexes[end_index_relative]]
        marked[candidate_indexes[end_index_relative]] = True
        neighbored_indexes = np.where(neighbored_flag[candidate_indexes[end_index_relative], :])[0]
        if len(neighbored_indexes) == 1:
            next_index = neighbored_indexes[0]  # 下一个点的序号
        else:
            next_index = neighbored_indexes[0] if marked[neighbored_indexes[1]] else neighbored_indexes[1]
        while not marked[next_index]:  # 依次访问该折线上的各个点，若下一个邻点且已经被标记，则表示来到了折线的端点，此时终止该折线
            neighbored_branch.append(next_index)
            marked[next_index] = True
            neighbored_indexes = np.where(neighbored_flag[next_index, :])[0]
            if len(neighbored_indexes) == 1:
                next_index = neighbored_indexes[0]  # 下一个点的序号
            else:
                next_index = neighbored_indexes[0] if marked[neighbored_indexes[1]] else neighbored_indexes[1]
        neighbored_groups.append(neighbored_branch)
    if marked[candidate_indexes].all():  # 所有考察点均已经标记，则结束
        return neighbored_groups, np.where(neighbored_numbers == 0)[0]
    # 存在仍然没有被标记的考察点，则认为其组成环形
    while not marked[candidate_indexes].all():  # 重复执行，直到所有考察点均已经标记
        neighbored_branch = []
        next_index = candidate_indexes[np.where(~marked[candidate_indexes])[0][0]]
        while not marked[next_index]:  # 依次访问该折线上的各个点，若下一个点已经被标记，则表明环线闭合
            neighbored_branch.append(next_index)
            marked[next_index] = True
            neighbored_indexes = np.where(neighbored_flag[next_index, :])[0]
            if len(neighbored_indexes) == 1:
                next_index = neighbored_indexes[0]  # 下一个点的序号
            else:
                next_index = neighbored_indexes[0] if marked[neighbored_indexes[1]] else neighbored_indexes[1]
        neighbored_groups.append(neighbored_branch)
    return neighbored_groups, np.where(neighbored_numbers == 0)[0]


def compress_sketch(graph, size=None, target_count=None, threshold=None):
    """
对所给的原始素描图进行压缩，通过舍弃部分冗余的拐点实现。
    :param graph: 原始素描图，字典，包含 positions 和 connections 两个元素
    :param size: 绘制素描图的画布大小
    :param target_count: 目标素描图中素描点的数目，一般而言数据量只与素描点的数目有关，连接关系部分影响不大。
    :param threshold: 用于拐点缩减的距离阈值。若阈值与目标数目同时出现，则以阈值为准。
    :return: 缩减后的素描图及绘制的边缘图像。
    """
    if size is None:
        size = [800, 800]
    threshold_connections = 0.3
    positions, connections = graph['positions'], graph['connections']
    # 首先舍弃不存在任何连接关系的孤立点
    degrees = (connections > threshold_connections).sum(axis=1)
    keep_indexes = np.where(degrees >= 1)[0]  # 存在连接关系的点
    positions = positions[keep_indexes, :]
    connections = connections[keep_indexes[None, :], keep_indexes[:, None]]
    if threshold is not None:  # 以阈值为标准进行筛选
        # 获取拐点序号和其邻点的序号，并计算拐点到其邻点连线的距离
        degrees = (connections > threshold_connections).sum(axis=1)  # 各个关键点的度
        corner_indexes = np.where(degrees == 2)[0]  # 拐点在所有关键点中的序号/位置
        if len(corner_indexes) != 0:
            corner_neighbors = np.where(connections[corner_indexes, :] > threshold_connections)[1].reshape(
                len(corner_indexes), 2)  # 各个拐点相邻点的序号（各个拐点）
            distances = np.array([get_distance_from_point_to_line(
                positions[corner_indexes[i]],
                positions[corner_neighbors[i, 0]],
                positions[corner_neighbors[i, 1]]) for i in range(len(corner_indexes))])  # 各个拐点到其相邻点连线的距离
            while np.min(distances) < threshold:
                # 对拐点按照距离升序排列，取出拐点作为考察点
                distances_sorted = np.sort(distances)
                indexes_sorted_relative = np.argsort(distances)  # 排序后拐点的序号（在所有拐点中的位置）
                indexes_sorted = corner_indexes[indexes_sorted_relative]  # 对各个拐点按照距离由低到高排序（排序后各个拐点在所有点中的位置）
                indexes_candidate_relative = indexes_sorted_relative[distances_sorted < threshold]  # 根据目标数目需要舍弃的拐点（在所有拐点中的位置）
                indexes_candidate = indexes_sorted[distances_sorted < threshold]  # 根据目标数目需要舍弃的拐点序号（在所有点中的序号）
                # 获取考察点之间的连接关系，并进行分组，每组互不连接
                neighbored = connections[indexes_candidate[None, :], indexes_candidate[:, None]] > threshold_connections  # 各个候选拐点是否相邻的标志
                candidate_groups_relative, candidate_isolate_relative = group_neighbored(neighbored)  # 找到存在邻接关系的候选点序号并对其进行分组，一组内是一条折线（拐点在所有候选点中的位置）
                candidate_groups = [indexes_candidate_relative[i] for i in candidate_groups_relative]  # 有连接的拐点在所有拐点中的位置
                candidate_isolate = indexes_candidate_relative[candidate_isolate_relative]  # 孤立拐点在所有拐点中的位置
                # 对各组考察点进行舍弃操作，记录需要舍弃的点和需要补充连接关系的点
                indexes2abandon, connections2makeup = list(corner_indexes[candidate_isolate]), [corner_neighbors[i] for i in candidate_isolate]
                for candidate_group in candidate_groups:
                    abandon_indexes = candidate_group[::2]
                    indexes2abandon += list(corner_indexes[abandon_indexes])
                    connections2makeup += [corner_neighbors[i] for i in abandon_indexes]
                # 执行关键点的舍弃及连接关系的补充
                connections2makeup = np.array(connections2makeup)
                connections[connections2makeup[:, 0], connections2makeup[:, 1]] = 1
                connections[connections2makeup[:, 1], connections2makeup[:, 0]] = 1
                indexes_reserved = np.array(list(set(range(positions.shape[0])).difference(set(indexes2abandon))))
                positions = positions[indexes_reserved, :]
                connections = connections[indexes_reserved[:, None], indexes_reserved[None, :]]
                degrees = (connections > threshold_connections).sum(axis=1)  # 各个关键点的度
                corner_indexes = np.where(degrees == 2)[0]  # 拐点在所有关键点中的序号/位置
                if len(corner_indexes) == 0:
                #     print('No more points to abandon! ')
                    break
                corner_neighbors = np.where(connections[corner_indexes, :] > threshold_connections)[1].reshape(
                    len(corner_indexes), 2)  # 各个拐点相邻点的序号（各个拐点）
                distances = np.array([get_distance_from_point_to_line(
                    positions[corner_indexes[i]],
                    positions[corner_neighbors[i, 0]],
                    positions[corner_neighbors[i, 1]]) for i in range(len(corner_indexes))])  # 各个拐点到其相邻点连线的距离
    else:
        while positions.shape[0] > target_count:  # 重复执行筛选，直到保留指定数目的关键点
            # 根据当前关键点的数目计算需要舍弃的关键点的数目N
            abandon_count = positions.shape[0] - target_count  # 需要舍弃的点的数目
            # 获取拐点序号和其邻点的序号，并计算拐点到其邻点连线的距离
            degrees = (connections > threshold_connections).sum(axis=1)  # 各个关键点的度
            corner_indexes = np.where(degrees == 2)[0]  # 拐点在所有关键点中的序号/位置
            if len(corner_indexes) == 0:
                print('No more points to abandon! ')
                break
            corner_neighbors = np.where(connections[corner_indexes, :] > threshold_connections)[1].reshape(len(corner_indexes), 2)  # 各个拐点相邻点的序号（各个拐点）
            distances = np.array([get_distance_from_point_to_line(
                positions[corner_indexes[i]],
                positions[corner_neighbors[i, 0]],
                positions[corner_neighbors[i, 1]]) for i in range(len(corner_indexes))])  # 各个拐点到其相邻点连线的距离
            # 对拐点按照距离升序排列，取出N个拐点作为考察点
            indexes_sorted_relative = np.argsort(distances)  # 排序后拐点的序号（在所有拐点中的位置）
            indexes_sorted = corner_indexes[indexes_sorted_relative]  # 对各个拐点按照距离由低到高排序（排序后各个拐点在所有点中的位置）
            indexes_candidate_relative = indexes_sorted_relative[:abandon_count]  # 根据目标数目需要舍弃的拐点（在所有拐点中的位置）
            indexes_candidate = indexes_sorted[:abandon_count]  # 根据目标数目需要舍弃的拐点序号（在所有点中的序号）
            # 获取考察点之间的连接关系，并进行分组，每组互不连接
            neighbored = connections[indexes_candidate[None, :], indexes_candidate[:, None]] > threshold_connections  # 各个候选拐点是否相邻的标志
            candidate_groups_relative, candidate_isolate_relative = group_neighbored(neighbored)  # 找到存在邻接关系的候选点序号并对其进行分组，一组内是一条折线（拐点在所有候选点中的位置）
            candidate_groups = [indexes_candidate_relative[i] for i in candidate_groups_relative]  # 有连接的拐点在所有拐点中的位置
            candidate_isolate = indexes_candidate_relative[candidate_isolate_relative]  # 孤立拐点在所有拐点中的位置
            # 对各组考察点进行舍弃操作，记录需要舍弃的点和需要补充连接关系的点
            indexes2abandon, connections2makeup = list(corner_indexes[candidate_isolate]), [corner_neighbors[i] for i in candidate_isolate]
            for candidate_group in candidate_groups:
                abandon_indexes = candidate_group[::2]
                indexes2abandon += list(corner_indexes[abandon_indexes])
                connections2makeup += [corner_neighbors[i] for i in abandon_indexes]
            # 执行关键点的舍弃及连接关系的补充
            connections2makeup = np.array(connections2makeup)
            connections[connections2makeup[:, 0], connections2makeup[:, 1]] = 1
            connections[connections2makeup[:, 1], connections2makeup[:, 0]] = 1
            indexes_reserved = np.array(list(set(range(positions.shape[0])).difference(set(indexes2abandon))))
            positions = positions[indexes_reserved, :]
            connections = connections[indexes_reserved[:, None], indexes_reserved[None, :]]
    graph = {'positions': positions,
             'connections': connections}
    sketch = plot_sketch(graph['positions'], size, graph['connections'], threshold_connection=threshold_connections)[0]
    return sketch, graph


def dec2bin(n, n_bit):
    """
    对十进制整数转化为二进制01字符串序列
    """
    return bin(n)[2:].rjust(n_bit, '0')


def bin2dec(bits):
    """
    对二进制字符串转化为十进制数
    """
    return int(bits, base=2)


def bitstring2bytes(bitstring='', makeup=True):
    """
    :param makeup: when length miss-matching, if we want to make up in the tail or abandon the tail.
    :param bitstring: string made up with '0'/'1'
    :return: bytes string
    """
    if makeup:
        bitstring_makeup = bitstring.ljust(int(np.ceil(len(bitstring) / 8)) * 8, '0')
    else:
        bitstring_makeup = bitstring[:int(np.floor(len(bitstring) / 8)) * 8]
    byte = b''
    for index in range(int(len(bitstring_makeup) / 8)):
        byte += np.array([int('0b' + bitstring_makeup[index * 8:(index + 1) * 8], 2)], 'uint8').tobytes()
    return byte


def bytes2bitstring(byte=b''):
    bitstring = ''
    for index in range(len(byte)):
        bitstring += bin(int(np.frombuffer(byte[index:index + 1], 'uint8')))[2:].rjust(8, '0')
    return bitstring


def rearrange_graph(graph):
    """
基于素描点中大部分点为拐点的特性对素描图进行压缩，其中包含以下步骤：
    1. 对关键点进行重排列，使得对应邻接矩阵的 1 集中到次对角线；
    2. 对重排列后的邻接矩阵与次对角线全为 1 的标准邻接矩阵做抑或操作；
    3. 对抑或操作后的邻接矩阵进行编码（即 save_graph_compressed）。
    3.1 对于断开的连接，抑或值为True的位置往往在次对角线，此时只需要编码一维坐标
    3.2 对于多余的连接，则需要对连接双方的序号进行编码
    此函数实现重排的功能
    """
    positions, connections = graph['positions'].copy(), graph['connections'].copy()
    connections_copy = connections.copy()
    # 进行重排列，基本思路为将具有顺序连接关系的关键点集中排列在一起
    threshold_connection = 0.3
    connections = connections > threshold_connection  # 各个关键点是否连接的标志
    N = positions.shape[0]
    indexes = []
    left_indexes = list(range(N))
    while len(indexes) < N:
        # Step 1: 计算各个关键点度
        degrees = connections.sum(axis=1)
        # Step 2: 选择一个端点或随机的一个点（若不存在端点）
        if 1 in degrees:
            index = np.where(degrees == 1)[0][0]
        else:
            index = left_indexes[0]
        # Step 3: 将所选点排列入栈
        indexes.append(index)
        left_indexes.remove(index)
        # Step 4: 找到所选点的邻点。若不存在，则直接进入下一个循环
        while connections[index].any():
            # Step 5: 找到所选点的邻点，将所选点连接关系置为0
            index_neighbor = int(np.where(connections[index])[0][0])
            if index_neighbor not in left_indexes:  # 该点的邻点已经访问过，说明遇到环路
                break
            connections[index, :] = 0
            connections[:, index] = 0
            index = index_neighbor
            indexes.append(index)
            left_indexes.remove(index)
        connections[index, :] = 0
        connections[:, index] = 0
    positions = positions[indexes, :]
    connections = connections_copy[np.array(indexes)[None, :], np.array(indexes)[:, None]]
    graph_rearranged = {'positions': positions, 'connections': connections}
    return graph_rearranged


def save_graph_compressed(graph, save_path="", size=None):
    """
对素描图以图的形式进行编码，码流组成为 (bit):
    [11 bit] 0~10: 图像的宽度W-1（表示0~2047，H同）
    [11 bit] 11~21: 图像的高度H-1
    [11 bit] 22~32: x的最小值（表示0~2047，y同）
    [11 bit] 33~43: y的最小值
    [ 4 bit] 44~47: X，对x的单个数值编码的比特数（表示0~15，y同）
    [ 4 bit] 48~51: Y，对y的单个数值编码的比特数
    [16 bit] 52~77: N，关键点的个数，M=ceil(log2(N))表示实际有效的位数，即对邻接矩阵坐标编码的比特数
    [ M bit] 78~77+M: C，异或结果在次对角线有值的个数
    [NX bit] 78+M~77+M+NX: 对N个点的横坐标x-x_min编码的结果
    [NY bit] 78+M+NX~77+M+NX+NY: 对N个点的纵坐标y-y_min编码的结果
    [CM bit] 78+M+NX+CM~77+M+NX+NY+CM: 对C个对角线有值的异或结果的单个坐标编码的结果
    [2QM bit] 78+M+NX+CM+2QM~77+M+NX+NY+CM+2QM: 对Q个其余异或结果的两个坐标编码的结果
    """
    if size is None:
        size = np.max(graph['positions'], axis=0)
    threshold_connection = 0.3
    positions, connections = graph['positions'].astype('int'), graph['connections'] > threshold_connection
    N = positions.shape[0]
    y_min, x_min = np.min(positions, axis=0)
    y_max, x_max = np.max(positions, axis=0)
    y_range, x_range = y_max - y_min + 1, x_max - x_min + 1
    h, w = size
    bit_x = int(np.ceil(np.log2(x_range)))  # 为x编码的比特数
    bit_y = int(np.ceil(np.log2(y_range)))  # 为y编码的比特数
    x_delta = positions[:, 1] - x_min
    y_delta = positions[:, 0] - y_min
    connections = np.triu(connections)
    matrix_standard = np.diag(np.ones((connections.shape[0] - 1,)), 1)  # 只有次对角线有值的基准矩阵
    connections_xor = np.logical_xor(connections, matrix_standard)  # 对连接关系矩阵与基准矩阵求抑或
    xor_positions = np.array(np.where(connections_xor)).transpose()  # 异或结果为1的位置
    sub_diag_flags = (xor_positions[:, 1] - xor_positions[:, 0]) == 1  # 异或结果为1的位置在次对角线的标志
    C = np.sum(sub_diag_flags)  # 异或结果在次对角线有值的个数
    sub_diag_pos = xor_positions[sub_diag_flags, 0]  # 异或结果在次对角线的坐标（行数，列数为行数加一）
    other_pos = xor_positions[~sub_diag_flags, :]  # 异或结果在其他位置的坐标
    M = int(np.ceil(np.log2(N)))  # 对一个连接关系坐标进行编码需要的比特数
    # 获取01码流
    bins_W = dec2bin(w-1, 11)
    bins_H = dec2bin(h - 1, 11)
    bins_x_min = dec2bin(x_min, 11)
    bins_y_min = dec2bin(y_min, 11)
    bins_x_bit = dec2bin(bit_x, 4)
    bins_y_bit = dec2bin(bit_y, 4)
    bins_N = dec2bin(N, 16)
    bins_C = dec2bin(C, M)
    bins_x_delta = [dec2bin(item, bit_x) for item in x_delta]
    bins_y_delta = [dec2bin(item, bit_y) for item in y_delta]
    # print("Points: ", len(''.join(bins_x_delta) + ''.join(bins_y_delta)))
    bins_sub_diag_pos = [dec2bin(item, M) for item in sub_diag_pos]
    bins_other_pos = [dec2bin(item, M) for item in other_pos.flatten()]
    bins = bins_W + bins_H + bins_x_min + bins_y_min + bins_x_bit + bins_y_bit \
        + bins_N + bins_C + ''.join(bins_x_delta) + ''.join(bins_y_delta) \
        + ''.join(bins_sub_diag_pos) + ''.join(bins_other_pos)
    # print("Total: ", len(bins))
    bytes_series = bitstring2bytes(bins)
    if save_path != "":
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, 'wb') as f:
            f.write(bytes_series)
    else:
        return bytes_series



def save_graph_compressed_bounding(graph, save_path="", size=None, branch_recorder=None):
    """
对素描图以图的形式进行编码，对坐标的编码利用分支的界定先验。码流组成为 (bit):
    [11 bit] 0~10: 图像的宽度W-1（表示0~2047，H同）
    [11 bit] 11~21: 图像的高度H-1
    [11 bit] 22~32: x的最小值（表示0~2047，y同）
    [11 bit] 33~43: y的最小值
    [ 4 bit] 44~47: X，对x的单个数值编码的比特数（表示0~15，y同）
    [ 4 bit] 48~51: Y，对y的单个数值编码的比特数
    [16 bit] 52~77: N，关键点的个数，M=ceil(log2(N))表示实际有效的位数，即对邻接矩阵坐标编码的比特数
    [ M bit] 78~77+M: C，异或结果在次对角线有值的个数。根据此可以得到交叉点、拐点和端点的个数，记为U,V,W
    [(U+W)X bit] 78+M~77+M+(U+W)X: 对U个交叉点和W个端点的横坐标x-x_min编码的结果
    [(U+W)Y bit] 78+M+(U+W)X~77+M+(U+W)X+(U+W)Y: 对N个点的纵坐标y-y_min编码的结果
    [CM bit] 78+M+(U+W)X+(U+W)Y~77+M+(U+W)X+(U+W)Y+CM: 对C个对角线有值的异或结果的单个坐标编码的结果
    [2QM bit] 78+M+(U+W)X+(U+W)Y+CM+2QM~77+M+(U+W)X+(U+W)Y+CM+2QM: 对Q个其余异或结果的两个坐标编码的结果
    [T bit] 对剩余所有V个拐点编码的结果
    """
    if size is None:
        size = np.max(graph['positions'], axis=0)
    threshold_connection = 0.3
    positions, connections = graph['positions'].astype('int'), graph['connections'] > threshold_connection
    N = positions.shape[0]
    degree = connections.sum(axis=0)
    flag_cross = degree >= 3
    flag_corner = degree == 2
    flag_end = degree == 1
    assert np.logical_or(flag_cross, flag_end).any(), "No cross points!"
    y_min, x_min = np.min(positions[np.logical_or(flag_cross, flag_end)], axis=0)
    y_max, x_max = np.max(positions[np.logical_or(flag_cross, flag_end)], axis=0)
    y_range, x_range = y_max - y_min + 1, x_max - x_min + 1
    h, w = size
    bit_x = int(np.ceil(np.log2(x_range)))  # 为x编码的比特数
    bit_y = int(np.ceil(np.log2(y_range)))  # 为y编码的比特数
    x_delta = positions[np.logical_or(flag_cross, flag_end)][:, 1] - x_min
    y_delta = positions[np.logical_or(flag_cross, flag_end)][:, 0] - y_min
    connections = np.triu(connections)
    matrix_standard = np.diag(np.ones((connections.shape[0] - 1,)), 1)  # 只有次对角线有值的基准矩阵
    connections_xor = np.logical_xor(connections, matrix_standard)  # 对连接关系矩阵与基准矩阵求抑或
    xor_positions = np.array(np.where(connections_xor)).transpose()  # 异或结果为1的位置
    sub_diag_flags = (xor_positions[:, 1] - xor_positions[:, 0]) == 1  # 异或结果为1的位置在次对角线的标志
    C = np.sum(sub_diag_flags)  # 异或结果在次对角线有值的个数
    sub_diag_pos = xor_positions[sub_diag_flags, 0]  # 异或结果在次对角线的坐标（行数，列数为行数加一）
    other_pos = xor_positions[~sub_diag_flags, :]  # 异或结果在其他位置的坐标
    M = int(np.ceil(np.log2(N)))  # 对一个连接关系坐标进行编码需要的比特数
    # 获取固定部分的01码流
    bins_W = dec2bin(w-1, 11)
    bins_H = dec2bin(h-1, 11)
    bins_x_min = dec2bin(x_min, 11)
    bins_y_min = dec2bin(y_min, 11)
    bins_x_bit = dec2bin(bit_x, 4)
    bins_y_bit = dec2bin(bit_y, 4)
    bins_N = dec2bin(N, 16)
    bins_C = dec2bin(C, M)
    bins_x_delta = [dec2bin(item, bit_x) for item in x_delta]
    bins_y_delta = [dec2bin(item, bit_y) for item in y_delta]
    bins_sub_diag_pos = [dec2bin(item, M) for item in sub_diag_pos]
    bins_other_pos = [dec2bin(item, M) for item in other_pos.flatten()]
    bins = bins_W + bins_H + bins_x_min + bins_y_min + bins_x_bit + bins_y_bit \
        + bins_N + bins_C + ''.join(bins_x_delta) + ''.join(bins_y_delta) \
        + ''.join(bins_sub_diag_pos) + ''.join(bins_other_pos)
    # print("total (cross+end): ", len(''.join(bins_x_delta) + ''.join(bins_y_delta)))
    # print("total (no corner):", len(bins))
    # 获取拐点的码流（各分支没有依赖关系，因此逐分支编码不丢失一般性）
    bins_corners = []
    for branch in branch_recorder.branches:
        bin_branch = ''
        start_pos = branch.start_pos
        end_pos = branch.end_pos
        mid_pos = branch.mid_pos
        all_pos = [start_pos] + mid_pos + [end_pos]
        if len(mid_pos) == 0:
            continue
        # 端点所能标记的矩形框范围
        y_min, y_max = (start_pos[0], end_pos[0]) if start_pos[0] < end_pos[0] else (end_pos[0], start_pos[0])
        x_min, x_max = (start_pos[1], end_pos[1]) if start_pos[1] < end_pos[1] else (end_pos[1], start_pos[1])
        # 分支所处矩形框范围
        ye_min, xe_min = np.array(all_pos).min(axis=0)
        ye_max, xe_max = np.array(all_pos).max(axis=0)
        # 矩形框向四个方向扩展的距离
        d_t, d_b = max(0, y_min - ye_min), max(0, ye_max - y_max)
        d_l, d_r = max(0, x_min - xe_min), max(0, xe_max - x_max)
        # 编码：向四个方向扩展距离编码的位数及距离
        bit_t = int(np.ceil(np.log2(np.ceil(np.log2(y_min + 1)) + 1)))  # 用这么多比特对向上扩展距离的编码位数进行编码
        if bit_t > 0:
            bin_t_b = dec2bin(int(np.ceil(np.log2(d_t + 1))), bit_t)  # 向上扩展距离编码位数的码流
        else:
            bin_t_b = ''
        if d_t > 0:
            bin_t = dec2bin(d_t, int(np.ceil(np.log2(d_t + 1))))[1:]  # 第一位一定是1，不编入码流
        else:
            bin_t = '0'
        bin_branch += bin_t_b + bin_t
        bit_l = int(np.ceil(np.log2(np.ceil(np.log2(x_min + 1)) + 1)))  # 用这么多比特对向上扩展距离的编码位数进行编码
        if bit_l > 0:
            bin_l_b = dec2bin(int(np.ceil(np.log2(d_l + 1))), bit_l)  # 向上扩展距离编码位数的码流
        else:
            bin_l_b = ''
        if d_l > 0:
            bin_l = dec2bin(d_l, int(np.ceil(np.log2(d_l + 1))))[1:]  # 第一位一定是1，不编入码流
        else:
            bin_l = '0'
        bin_branch += bin_l_b + bin_l
        bit_b = int(np.ceil(np.log2(np.ceil(np.log2(h - y_max)) + 1)))  # 用这么多比特对向上扩展距离的编码位数进行编码
        if bit_b > 0:
            bin_b_b = dec2bin(int(np.ceil(np.log2(d_b + 1))), bit_b)  # 向上扩展距离编码位数的码流
        else:
            bin_b_b = ''
        if d_b > 0:
            bin_b = dec2bin(d_b, int(np.ceil(np.log2(d_b + 1))))[1:]  # 第一位一定是1，不编入码流
        else:
            bin_b = '0'
        bin_branch += bin_b_b + bin_b
        bit_r = int(np.ceil(np.log2(np.ceil(np.log2(w - x_max)) + 1)))  # 用这么多比特对向上扩展距离的编码位数进行编码
        if bit_r > 0:
            bin_r_b = dec2bin(int(np.ceil(np.log2(d_r + 1))), bit_r)  # 向上扩展距离编码位数的码流
        else:
            bin_r_b = ''
        if d_r > 0:
            bin_r = dec2bin(d_r, int(np.ceil(np.log2(d_r + 1))))[1:]  # 第一位一定是1，不编入码流
        else:
            bin_r = '0'
        bin_branch += bin_r_b + bin_r
        # 编码：中间点的坐标
        bit_x = int(np.ceil(np.log2(xe_max - xe_min + 1)))  # 对 x 编码的比特数
        bit_y = int(np.ceil(np.log2(ye_max - ye_min + 1)))  # 对 y 编码的比特数
        x_diffs = [pos[1] - xe_min for pos in mid_pos]
        y_diffs = [pos[0] - ye_min for pos in mid_pos]
        x_codes = [dec2bin(item, bit_x) for item in x_diffs]
        y_codes = [dec2bin(item, bit_y) for item in y_diffs]
        bin_branch += ''.join(x_codes) + ''.join(y_codes)
        bins_corners.append(bin_branch)
        # print(len(bin_branch), len(mid_pos), len(bin_branch) / len(mid_pos))
    bins += ''.join(bins_corners)
    # print("Total (corners): ", len(''.join(bins_corners)))
    # print("Total: ", len(bins))
    bytes_series = bitstring2bytes(bins)
    if save_path != "":
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, 'wb') as f:
            f.write(bytes_series)
    else:
        return bytes_series


def k2tree(connections: np.ndarray):
    # STEP 1: 判断是否是2的次幂。如果不是，则补零
    n = connections.shape[0]
    if n & (n - 1) > 0:  # 不是2的次幂
        # 补充维度为2的次幂
        N = int(2 ** np.ceil(np.log2(n)))
        connections_padded = np.pad(connections, ((0, N - n), (0, N - n)))
    else:
        N = n
        connections_padded = connections

    # STEP 2: 定义核心算法：分块、返回4个块是否有1

    def get_code(matrix):
        n_local = matrix.shape[0]
        if n_local == 1:
            return matrix.any()
        return [int(matrix[:int(n_local/2), :int(n_local/2)].any()),
                int(matrix[:int(n_local/2), int(n_local/2):].any()),
                int(matrix[int(n_local/2):, :int(n_local/2)].any()),
                int(matrix[int(n_local/2):, int(n_local/2):].any())]

    # STEP 3: 迭代，获取码流
    if N == 1:
        code = []
    elif connections_padded.any():
        code = get_code(connections_padded)
        code += k2tree(connections_padded[:int(N/2), :int(N/2)])
        code += k2tree(connections_padded[:int(N/2), int(N/2):])
        code += k2tree(connections_padded[int(N/2):, :int(N/2)])
        code += k2tree(connections_padded[int(N/2):, int(N/2):])
    else:
        code = []
    return code


def repair(connections: np.ndarray):
    T = []
    dictionary = dict()  # 键为原始对构成的字符串，如'3,4'，值为替换的新值，如'7'
    dictionary_reverse = dict()  # 键值对含义相反，但键的类型为int，值为列表
    N = connections.shape[0]
    new_index = N

    def count(lists):
        """统计出现的连续数字的频率"""
        collections_pair = []
        for item in lists:
            if len(item) == 1:
                continue
            for index in range(len(item) - 1):
                collections_pair.append(','.join([str(n) for n in item[index:index + 2]]))
        counts = dict(collections.Counter(collections_pair))
        return counts

    # STEP 1: 获取邻接表
    for i in range(N):
        T.append(list(np.where([connections[i]])[1]))
    # STEP 2: 统计点对出现的频率，将词频最高者记入词典
    while True:
        pair_counts = count(T)
        frequencies = pair_counts.values()
        if frequencies.__len__() == 0:
            break
        if max(frequencies) == 1:
            break
        max_index = int(np.argmax(list(frequencies)))  # 出现频率最高的序号
        pair_ori = list(pair_counts.keys())[max_index]
        dictionary.update({pair_ori: new_index})
        dictionary_reverse.update({new_index: [int(i) for i in pair_ori.split(',')]})
        # STEP 3: 替换T中的相关元素
        for T_sub in T:
            T_sub_str = ','.join([str(item) for item in T_sub])
            if pair_ori in T_sub_str:  # 这一行中有待替换元素
                str_index = T_sub_str.index(pair_ori)
                index_replace = T_sub_str[:str_index].count(',')
                # index_replace = int(T_sub_str.index(pair_ori)/2)  # 替换的位置
                T_sub.pop(index_replace)
                T_sub.pop(index_replace)
                T_sub.insert(index_replace, new_index)
        # STEP 4: 新元素序号加1
        new_index += 1
    # STEP 5: 组织码流
    """码流组织：（假设总点数已经编码过）
    [N*M bit] 0~MN-1: N个节点在最终邻接表中各自所邻接的点数，总和为P。每个数目范围为0~N，用M bit编码，共N个点
    [M bit] MN~MN+M-1: 字典的项数L。最多进行N-1次缩减，最多有N-1项，用M bit编码
    [M+1 bit] MN+M~MN+2M: T中出现的最大序号C∈[N-1~2N-2]。最多进行N-1次缩减出现序号2N-2，用M+1 bit编码。D=ceil(log2(C))，表示实际有效的位数，对后面的序号编码需要花费D bit
    [3DL bit] MN+2M+1~MN+2M+3DL: 编码字典。每一项需要编码原序号对和新序号，各花费D bit
    [PD bit] MN+2M+3DL+1~MN+2M+3DL+PD: 编码邻接表，共P个数，每个数用D bit编码
    """
    M = int(np.ceil(np.log2(N)))  # 对一个原始序号编码花费的比特数
    lengths = [len(T_sub) for T_sub in T]  # 各个节点的邻接点数
    P = sum(lengths)  # 邻接表中元素总个数
    bins_neighbor_points = [dec2bin(l_, M) for l_ in lengths]  # 对各个节点邻接数目的码流
    L = dictionary.__len__()  # 字典的项数
    bin_L = dec2bin(L, M)  # 字典的项数码流
    if dictionary.__len__() > 0:
        C = max(dictionary_reverse.keys())  # 最大序号
    else:
        C = N - 1
    D = int(np.ceil(np.log2(C+1)))  # 对一个现在的序号编码花费的比特数，可能是M或M+1
    bin_C = dec2bin(C, M+1)  # 最大序号的码流
    bins_dict = []
    for key in dictionary.keys():
        idx1, idx2 = key.split(',')
        bins_item = dec2bin(int(idx1), D) + dec2bin(int(idx2), D) + dec2bin(dictionary[key], D)
        bins_dict.append(bins_item)
    bins_neighbors = ''
    for t1 in T:
        for t2 in t1:
            bins_neighbors += dec2bin(t2, D)
    code = ''.join(bins_neighbor_points) + bin_L + bin_C + ''.join(bins_dict) + bins_neighbors
    return code


def save_graph_k2tree(graph, save_path, size):
    """
对素描图以图的形式进行编码，其中在对connections编码时常采用k2tree算法，码流组成为 (bit):
    [11 bit] 0~10: 图像的宽度W-1（表示0~2047，H同）
    [11 bit] 11~21: 图像的高度H-1
    [11 bit] 22~32: x的最小值（表示0~2047，y同）
    [11 bit] 33~43: y的最小值
    [ 4 bit] 44~47: X，对x的单个数值编码的比特数（表示0~15，y同）
    [ 4 bit] 48~51: Y，对y的单个数值编码的比特数
    [16 bit] 52~77: N，关键点的个数，M=ceil(log2(N))表示实际有效的位数，即对邻接矩阵坐标编码的比特数
    [NX bit] 78~77+NX: 对N个点的横坐标x-x_min编码的结果
    [NY bit] 78+NX~77+NX+NY: 对N个点的纵坐标y-y_min编码的结果
    [X bit] connections 编码结果
    """
    threshold_connection = 0.3
    positions, connections = graph['positions'].astype('int'), graph['connections'] > threshold_connection
    N = positions.shape[0]
    y_min, x_min = np.min(positions, axis=0)
    y_max, x_max = np.max(positions, axis=0)
    y_range, x_range = y_max - y_min + 1, x_max - x_min + 1
    h, w = size
    bit_x = int(np.ceil(np.log2(x_range)))  # 为x编码的比特数
    bit_y = int(np.ceil(np.log2(y_range)))  # 为y编码的比特数
    x_delta = positions[:, 1] - x_min
    y_delta = positions[:, 0] - y_min
    connections = np.triu(connections)
    # 获取01码流
    bins_W = dec2bin(w - 1, 11)
    bins_H = dec2bin(h - 1, 11)
    bins_x_min = dec2bin(x_min, 11)
    bins_y_min = dec2bin(y_min, 11)
    bins_x_bit = dec2bin(bit_x, 4)
    bins_y_bit = dec2bin(bit_y, 4)
    bins_N = dec2bin(N, 16)
    bins_x_delta = [dec2bin(item, bit_x) for item in x_delta]
    bins_y_delta = [dec2bin(item, bit_y) for item in y_delta]
    bins_connections = k2tree(connections)
    bins = bins_W + bins_H + bins_x_min + bins_y_min + bins_x_bit + bins_y_bit \
           + bins_N + ''.join(bins_x_delta) + ''.join(bins_y_delta) \
           + ''.join([str(i) for i in bins_connections])
    bytes_series = bitstring2bytes(bins)
    with open(save_path, 'wb') as f:
        f.write(bytes_series)


def save_graph_repair(graph, save_path, size):
    """
对素描图以图的形式进行编码，其中在对connections编码时采用repair算法，码流组成为 (bit):
    [11 bit] 0~10: 图像的宽度W-1（表示0~2047，H同）
    [11 bit] 11~21: 图像的高度H-1
    [11 bit] 22~32: x的最小值（表示0~2047，y同）
    [11 bit] 33~43: y的最小值
    [ 4 bit] 44~47: X，对x的单个数值编码的比特数（表示0~15，y同）
    [ 4 bit] 48~51: Y，对y的单个数值编码的比特数
    [16 bit] 52~77: N，关键点的个数，M=ceil(log2(N))表示实际有效的位数，即对邻接矩阵坐标编码的比特数
    [NX bit] 78~77+NX: 对N个点的横坐标x-x_min编码的结果
    [NY bit] 78+NX~77+NX+NY: 对N个点的纵坐标y-y_min编码的结果
    [X bit] connections 编码结果
    """
    threshold_connection = 0.3
    positions, connections = graph['positions'].astype('int'), graph['connections'] > threshold_connection
    N = positions.shape[0]
    y_min, x_min = np.min(positions, axis=0)
    y_max, x_max = np.max(positions, axis=0)
    y_range, x_range = y_max - y_min + 1, x_max - x_min + 1
    h, w = size
    bit_x = int(np.ceil(np.log2(x_range)))  # 为x编码的比特数
    bit_y = int(np.ceil(np.log2(y_range)))  # 为y编码的比特数
    x_delta = positions[:, 1] - x_min
    y_delta = positions[:, 0] - y_min
    connections = np.triu(connections)
    # 获取01码流
    bins_W = dec2bin(w - 1, 11)
    bins_H = dec2bin(h - 1, 11)
    bins_x_min = dec2bin(x_min, 11)
    bins_y_min = dec2bin(y_min, 11)
    bins_x_bit = dec2bin(bit_x, 4)
    bins_y_bit = dec2bin(bit_y, 4)
    bins_N = dec2bin(N, 16)
    bins_x_delta = [dec2bin(item, bit_x) for item in x_delta]
    bins_y_delta = [dec2bin(item, bit_y) for item in y_delta]
    bins_connections = repair(connections)
    bins = bins_W + bins_H + bins_x_min + bins_y_min + bins_x_bit + bins_y_bit \
           + bins_N + ''.join(bins_x_delta) + ''.join(bins_y_delta) \
           + ''.join([str(i) for i in bins_connections])
    bytes_series = bitstring2bytes(bins)
    with open(save_path, 'wb') as f:
        f.write(bytes_series)


def decode_graph(input_):
    """
    解码素描图码流。
    :param input_: 输入，素描图编码后的文件（字符串）或字节流（字节）
    :return: 图像尺寸（[H,W]），素描图（graph），与绘制的素描线（sketch）
    """
    if input_.__class__ == str:
        with open(input_, 'rb') as f:
            bytes_ = f.read()
        bins = bytes2bitstring(bytes_)
    else:
        assert input_.__class__ == bytes
        bins = input_
    bins_size = len(bins)
    start = 0  # 下一个信息的起始比特序号

    def decode_single():
        return bin2dec(bins[start:start + length])

    # 图像宽度
    length = 11
    W = decode_single() + 1
    start += length
    # 图像高度
    length = 11
    H = decode_single() + 1
    start += length
    # x 的最小值
    length = 11
    x_min = decode_single()
    start += length
    # y 的最小值
    length = 11
    y_min = decode_single()
    start += length
    # 对x的数值编码的比特数
    length = 4
    X = decode_single()
    start += length
    # 对y的数值编码的比特数
    length = 4
    Y = decode_single()
    start += length
    # 关键点的个数。M=ceil(log2(N))表示实际有效的位数，即对邻接矩阵坐标编码的比特数
    length = 16
    N = decode_single()
    start += length
    M = int(np.ceil(np.log2(N)))
    # 异或结果在次对角线有值的个数
    length = M
    C = decode_single()
    start += length
    # N个点的横坐标
    length = N * X
    x_errors = np.array([int(bins[start + i * X:start + (i + 1) * X], base=2) for i in range(N)])
    x = x_min + x_errors
    start += length
    # N个点的纵坐标
    length = N * Y
    y_errors = np.array([int(bins[start + i * Y:start + (i + 1) * Y], base=2) for i in range(N)])
    y = y_min + y_errors
    start += length
    # 次对角线异或值为1的位置（行数）
    length = C * M
    sub_dialog_pos = np.array([int(bins[start + i * M:start + (i + 1) * M], base=2) for i in range(C)])
    start += length
    # 其余位置异或值为1的坐标（行列式）。虽然在码流结尾补0时会导致译码歧义，但由于编码时码流为(行,列,行,列)交替的形式，而列不可能为0，因此可以实现正确解码。
    other_pos = np.empty([0, 2])
    length = M  # 一个坐标需要的比特数
    while True:
        if bins_size - start < 2 * length:
            print('decode done, with %d redundant bits.' % (bins_size - start))
            break
        line = decode_single()
        start += length
        column = decode_single()
        start += length
        if column == 0:
            print('decode done, with %d redundant bits.' % (bins_size - start + 2 * length))
            break
        other_pos = np.concatenate([other_pos, np.array([[line, column]])], axis=0)
    other_pos = other_pos.astype('int')
    # 整合到素描图
    positions = np.concatenate([y[:, None], x[:, None]], axis=1)
    ref_matrix = np.diag(np.ones((N - 1,)), 1)  # 只有次对角线有值的基准矩阵
    ref_matrix[sub_dialog_pos, sub_dialog_pos + 1] = 1 - ref_matrix[sub_dialog_pos, sub_dialog_pos + 1]  # 修正次对角线的值
    ref_matrix[other_pos[:, 0], other_pos[:, 1]] = 1 - ref_matrix[other_pos[:, 0], other_pos[:, 1]]  # 修正其他位置的值
    ref_matrix += ref_matrix.transpose()  # 对称的邻接矩阵
    graph = {'positions': positions, 'connections': ref_matrix}
    # 恢复素描图
    sketch = plot_sketch(graph['positions'], [H, W], connections=graph['connections'])[0]
    # cv2.imwrite('temp_replot.png', sketch[:, :, ::-1])
    return [H, W], graph, sketch[:, :, 0]


class Block:
    def __init__(self, indexes):
        self.l_start, self.c_start, self.l_end, self.c_end = indexes
        self.size = self.c_end - self.c_start + 1
        if self.size > 1:
            half_size = int(self.size/2)
            self.sub_block = [Block([self.l_start, self.c_start, self.l_start+half_size-1, self.c_start+half_size-1]),
                              Block([self.l_start, self.c_start+half_size, self.l_start+half_size-1, self.c_end]),
                              Block([self.l_start+half_size, self.c_start, self.l_end, self.c_start+half_size-1]),
                              Block([self.l_start+half_size, self.c_start+half_size, self.l_end, self.c_end])]


def walk_tree(connections, block, bins):
    candidate_blocks = []
    for i in range(4):
        if bins[i] == '0':
            connections[block.sub_block[i].l_start:block.sub_block[i].l_end+1,
                        block.sub_block[i].c_start:block.sub_block[i].c_end+1] = 0
        else:
            if block.sub_block[i].size == 1:
                connections[block.sub_block[i].l_start, block.sub_block[i].c_start] = 1
            else:
                candidate_blocks.append(block.sub_block[i])
    bins = bins[4:]
    if block.size > 2:
        for sub_block in candidate_blocks:
            bins = walk_tree(connections, sub_block, bins)
    return bins


def decode_k2tree(bins, N):
    """
    k2tree图压缩算法的解码函数
    :param bins: 由'0'和'1'组成的字符串
    :param N: 实际节点数，即实际有效的邻接矩阵的尺寸
    :return: 解码后的邻接矩阵，其尺寸可能超过实际尺寸，需要结合实际的节点数进行切片
    """
    N_expand = int(2 ** np.ceil(np.log2(N)))
    connections = np.ones([N_expand, N_expand]) * -1
    block = Block([0, 0, N_expand-1, N_expand-1])
    walk_tree(connections, block, list(bins))
    return connections[:N, :N]


def decode_graph_k2tree(input_):
    """
    解码素描图码流。
    :param input_: 输入，素描图编码后的文件（字符串）或字节流（字节）
    :return: 图像尺寸（[H,W]），素描图（graph），与绘制的素描线（sketch）
    """
    if input_.__class__ == str:
        with open(input_, 'rb') as f:
            bytes_ = f.read()
        bins = bytes2bitstring(bytes_)
    else:
        assert input_.__class__ == bytes
        bins = input_
    bins_size = len(bins)
    start = 0  # 下一个信息的起始比特序号

    def decode_single():
        return bin2dec(bins[start:start + length])

    # 图像宽度
    length = 11
    W = decode_single() + 1
    start += length
    # 图像高度
    length = 11
    H = decode_single() + 1
    start += length
    # x 的最小值
    length = 11
    x_min = decode_single()
    start += length
    # y 的最小值
    length = 11
    y_min = decode_single()
    start += length
    # 对x的数值编码的比特数
    length = 4
    X = decode_single()
    start += length
    # 对y的数值编码的比特数
    length = 4
    Y = decode_single()
    start += length
    # 关键点的个数。M=ceil(log2(N))表示实际有效的位数，即对邻接矩阵坐标编码的比特数
    length = 16
    N = decode_single()
    start += length
    # M = int(np.ceil(np.log2(N)))
    # N个点的横坐标
    length = N * X
    x_errors = np.array([int(bins[start + i * X:start + (i + 1) * X], base=2) for i in range(N)])
    x = x_min + x_errors
    start += length
    # N个点的纵坐标
    length = N * Y
    y_errors = np.array([int(bins[start + i * Y:start + (i + 1) * Y], base=2) for i in range(N)])
    y = y_min + y_errors
    start += length
    # 解码邻接矩阵
    connections = decode_k2tree(bins[start:], N)
    # 整合到素描图
    positions = np.concatenate([y[:, None], x[:, None]], axis=1)
    graph = {'positions': positions, 'connections': connections + connections.transpose()}
    # 恢复素描图
    sketch = plot_sketch(graph['positions'], [H, W], connections=graph['connections'])[0]
    return [H, W], graph, sketch[:, :, 0]


def decode_repair(bins, N):

    M = int(np.ceil(np.log2(N)))
    start = 0

    def decode_single():
        number = bin2dec(bins[start:start + length])
        return number

    # 解码各个点在最终邻接表T中的邻点数目
    neighbor_count = []
    length = M
    for i in range(N):
        neighbor_count.append(decode_single())
        start += length
    neighbor_count = np.array(neighbor_count)
    P = sum(neighbor_count)
    # 解码字典的项数 L
    length = M
    L = decode_single()
    start += length
    # 解码T中出现的最大序号C
    length = M+1
    C = decode_single()
    D = int(np.ceil(np.log2(C+1)))
    start += length
    # 解码字典
    dictionary_reverse = dict()  # key为编码时映射后的序号，value为原始序号对
    length = D
    for i in range(L):
        index_1 = decode_single()
        start += length
        index_2 = decode_single()
        start += length
        index_new = decode_single()
        start += length
        dictionary_reverse.update({index_new: (index_1, index_2)})
    # 解码邻接矩阵
    length = D
    T = []
    for i in range(P):
        T.append(decode_single())
        start += length
    # 恢复邻接表
    start_index = np.cumsum([0] + list(neighbor_count))  # 各个节点在T中对应邻点的起始序号
    for merge_index in range(C, N-1, -1):  # 遍历新序号，从C到N
        replace_index = dictionary_reverse[merge_index]  # 新序号对应的原序号
        locations = np.argwhere(np.array(T) == merge_index)  # 新序号在T中的位置
        belongs = [np.argwhere(start_index > location)[0]-1 for location in locations]  # 新序号作为邻点所属的节点
        neighbor_count[np.array(belongs)] += 1  # 所属节点的邻点数目加一
        start_index = np.cumsum([0] + list(neighbor_count))  # 更新起始序号
        # 还原一步邻接矩阵
        revise_index = 0
        for location in locations:
            T.pop(int(location)+revise_index)
            T.insert(int(location)+revise_index, replace_index[1])
            T.insert(int(location)+revise_index, replace_index[0])
            revise_index += 1
    # 恢复邻接矩阵
    connections_recover = np.zeros([N, N])
    index_T = 0  # 邻点序号在邻接表T中的位置
    for n in range(N):
        for neighbor_index in range(neighbor_count[n]):
            connections_recover[n, T[index_T]] = 1
            index_T += 1
    return connections_recover


def decode_graph_repair(input_):
    """
    解码素描图码流。
    :param input_: 输入，素描图编码后的文件（字符串）或字节流（字节）
    :return: 图像尺寸（[H,W]），素描图（graph），与绘制的素描线（sketch）
    """
    if input_.__class__ == str:
        with open(input_, 'rb') as f:
            bytes_ = f.read()
        bins = bytes2bitstring(bytes_)
    else:
        assert input_.__class__ == bytes
        bins = input_
    bins_size = len(bins)
    start = 0  # 下一个信息的起始比特序号

    def decode_single():
        return bin2dec(bins[start:start + length])

    # 图像宽度
    length = 11
    W = decode_single() + 1
    start += length
    # 图像高度
    length = 11
    H = decode_single() + 1
    start += length
    # x 的最小值
    length = 11
    x_min = decode_single()
    start += length
    # y 的最小值
    length = 11
    y_min = decode_single()
    start += length
    # 对x的数值编码的比特数
    length = 4
    X = decode_single()
    start += length
    # 对y的数值编码的比特数
    length = 4
    Y = decode_single()
    start += length
    # 关键点的个数。M=ceil(log2(N))表示实际有效的位数，即对邻接矩阵坐标编码的比特数
    length = 16
    N = decode_single()
    start += length
    # M = int(np.ceil(np.log2(N)))
    # N个点的横坐标
    length = N * X
    x_errors = np.array([int(bins[start + i * X:start + (i + 1) * X], base=2) for i in range(N)])
    x = x_min + x_errors
    start += length
    # N个点的纵坐标
    length = N * Y
    y_errors = np.array([int(bins[start + i * Y:start + (i + 1) * Y], base=2) for i in range(N)])
    y = y_min + y_errors
    start += length
    # 解码邻接矩阵
    connections = decode_repair(bins[start:], N)
    # 整合到素描图
    positions = np.concatenate([y[:, None], x[:, None]], axis=1)
    graph = {'positions': positions, 'connections': connections + connections.transpose()}
    # 恢复素描图
    sketch = plot_sketch(graph['positions'], [H, W], connections=graph['connections'])[0]
    return [H, W], graph, sketch[:, :, 0]


def analysis_branch_prior(branch_object, assigned_index, ssm_regions, box_labels):
    """根据区域对应的分支序号信息进行素描图的编码，将各区域的矩形框作为已知信息"""
    if len(box_labels) == 0:
        return 0  # 没有矩形框，不需要进行任何处理
    bit_total = 0
    processed_branch = []  # 记录处理过的分支序号
    for region in ssm_regions:
        bit_box = int(np.ceil(np.log2(len(box_labels))))  # 记录区域对应矩形框
        branch_idx_list, label = region
        # 在 recorder 中实际对应的序号和路径，获取矩形框
        branch_idx_recorder = [assigned_index.index(i) for i in branch_idx_list]
        pos_region = []
        for branch_idx in branch_idx_recorder:
            pos_region += [branch_object.branches[branch_idx].start_pos] + branch_object.branches[branch_idx].mid_pos + [branch_object.branches[branch_idx].end_pos]
        pos_region = np.array(pos_region)
        min_l, min_c = np.min(pos_region, axis=0)
        max_l, max_c = np.max(pos_region, axis=0)
        bit_l = int(np.ceil(np.log2(max_l - min_l + 1)))
        bit_c = int(np.ceil(np.log2(max_c - min_c + 1)))
        # 考察未处理过的分支
        unprocessed_branch = [branch_idx for branch_idx in branch_idx_list if branch_idx not in processed_branch]
        processed_branch += unprocessed_branch
        bit_pos = 0
        for branch_idx in unprocessed_branch:
            branch = branch_object.branches[assigned_index.index(branch_idx)]
            bit_pos += (len(branch.mid_pos) + 2) * (bit_l + bit_c)
        bit_region = bit_box + bit_pos
        bit_total += bit_region
    return bit_total

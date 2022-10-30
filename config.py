
import os
import re

MAIN_DIR = ""

DATA_DIR = f'{MAIN_DIR}'
VOCAB_DIR = f'{MAIN_DIR}/vocab'
CKPT_DIR = f'{MAIN_DIR}/ckpt'
RANKING_DIR = f'{MAIN_DIR}/rankings'
HEATMAP_DIR = f'{MAIN_DIR}/heatmaps'


TORCH_HOME = ""
GLOVE_DIR = ""




FASHIONIQ_IMAGE_DIR = f''
FASHIONIQ_ANNOTATION_DIR = f''


SHOES_IMAGE_DIR = f''
SHOES_ANNOTATION_DIR = f''


CIRR_IMAGE_DIR = f''
CIRR_ANNOTATION_DIR = f''


FASHION200K_IMAGE_DIR = f''
FASHION200K_ANNOTATION_DIR = f''


cleanCaption = lambda cap : " ".join(re.sub('[^(a-zA-Z)\ ]', '', re.sub('[/\-\\\\]', ' ', cap)).split(" "))

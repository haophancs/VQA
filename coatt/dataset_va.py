import os
import operator
import numpy as np

from six.moves import cPickle as pickle
from collections import defaultdict
from external.vqa.vqa import VQA

image_dir = "/content/VQA/datasets/vivqa/val"
img_prefix = ""
qjson = "/content/VQA/datasets/vivqa/vqa/vivqa_val_questions.json"
ajson = "/content/VQA/datasets/vivqa/vqa/vivqa_val_annotations.json"

with open('/content/VQA/outputs/coatt/a2i.pkl', 'rb') as f:
    a2i = pickle.load(f)

vqa = VQA(ajson, qjson)

img_names = [f for f in os.listdir(image_dir) if '.jpg' in f]
img_ids = []
for fname in img_names:
    img_id = fname.split('.')[0]
    img_ids.append(int(img_id))

ques_ids = vqa.getQuesIds(img_ids)

ques_ids_modif = []
for ques_id in ques_ids:
    qa = vqa.loadQA(ques_id)[0]
    qqa = vqa.loadQQA(ques_id)[0]

    ques = qqa['question'][:-1]
    answers = qa['answers']
    answer = ""
    for ans in answers:
        ans = ans['answer'].lower()
        if ans in a2i:
            answer = ans
            break

    if not answer == "":
        ques_ids_modif.append(ques_id)

np.save('/content/VQA/outputs/coatt/va_img_names.npy', img_names)
np.save('/content/VQA/outputs/coatt/va_img_ids.npy', img_ids)
np.save('/content/VQA/outputs/coatt/va_ques_ids.npy', ques_ids_modif)

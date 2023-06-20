import os
import operator
import numpy as np

from six.moves import cPickle as pickle
from collections import defaultdict
from external.vqa.vqa import VQA

image_dir = "./datasets/viclevr/train"
img_prefix = ""
qjson = "./datasets/viclevr/vqa/viclevr_train_questions.json"
ajson = "./datasets/viclevr/vqa/viclevr_train_annotations.json"

vqa = VQA(ajson, qjson)

img_names = [f for f in os.listdir(image_dir) if '.jpg' in f]
img_ids = []
for fname in img_names:
    img_id = fname.split('.')[0].split('_')[-1]
    img_ids.append(int(img_id))

ques_ids = vqa.getQuesIds(img_ids)

q2i = defaultdict(lambda: len(q2i))
pad = q2i["<pad>"]
start = q2i["<sos>"]
end = q2i["<eos>"]
UNK = q2i["<unk>"]

a2i_count = {}
for ques_id in ques_ids:
    qa = vqa.loadQA(ques_id)[0]
    qqa = vqa.loadQQA(ques_id)[0]

    ques = qqa['question'][:-1]
    [q2i[x] for x in ques.lower().strip().split(" ")]

    answers = qa['answers']
    for ans in answers:
        ans = ans['answer'].lower()
        if ans not in a2i_count:
            a2i_count[ans] = 1
        else:
            a2i_count[ans] = a2i_count[ans] + 1

a_sort = sorted(a2i_count.items(), key=operator.itemgetter(1), reverse=True)

i2a = {}
count = 0
a2i = defaultdict(lambda: len(a2i))
for word, _ in a_sort:
    a2i[word]
    i2a[a2i[word]] = word
    count = count + 1
    if count == 1000:
        break

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

    if answer == "":
        continue
    ques_ids_modif.append(ques_id)

print(len(ques_ids_modif), len(ques_ids))
with open('./outputs/coatt/q2i.pkl', 'wb') as f:
    pickle.dump(dict(q2i), f)
with open('./outputs/coatt/a2i.pkl', 'wb') as f:
    pickle.dump(dict(a2i), f)
with open('./outputs/coatt/i2a.pkl', 'wb') as f:
    pickle.dump(i2a, f)
with open('./outputs/coatt/a2i_count.pkl', 'wb') as f:
    pickle.dump(a2i_count, f)

np.save('./outputs/coatt/q2i.npy', dict(q2i))
np.save('./outputs/coatt/a2i.npy', dict(a2i))
np.save('./outputs/coatt/i2a.npy', i2a)
np.save('./outputs/coatt/a2i_count.npy', a2i_count)

np.save('./outputs/coatt/tr_img_names.npy', img_names)
np.save('./outputs/coatt/tr_img_ids.npy', img_ids)
np.save('./outputs/coatt/tr_ques_ids.npy', ques_ids_modif)

import os
import pickle
import operator
import numpy as np

from collections import defaultdict
from external.vqa.vqa import VQA


def pre_process_dataset(image_dirs, qjsons, ajsons, img_prefix):
    print('Preprocessing datatset. \n')

    img_ids = []

    q2i = defaultdict(lambda: len(q2i))
    a2i_count = {}

    pad = q2i["<pad>"]
    start = q2i["<sos>"]
    end = q2i["<eos>"]
    UNK = q2i["<unk>"]

    for image_dir, qjson, ajson in zip(image_dirs, qjsons, ajsons):
        vqa = VQA(ajson, qjson)

        img_names = [f for f in os.listdir(image_dir) if '.png' in f]
        for fname in img_names:
            img_id = fname.split('.')[0].split('_')[-1]
            img_ids.append(int(img_id))

        for ques_id in vqa.getQuesIds(img_ids):
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

    return tuple(map(dict, (q2i, a2i, i2a, a2i_count)))


if __name__ == '__main__':
    image_dirs = ["./datasets/vigqa/train", "./datasets/vigqa/val", "./datasets/vigqa/test"]
    img_prefix = ""
    qjsons = [
        "./datasets/vigqa/vqa/vigqa_train_questions.json",
        "./datasets/vigqa/vqa/vigqa_val_questions.json",
        "./datasets/vigqa/vqa/vigqa_test_questions.json",
    ]
    ajsons = [
        "./datasets/vigqa/vqa/vigqa_train_annotations.json",
        "./datasets/vigqa/vqa/vigqa_val_annotations.json",
        "./datasets/vigqa/vqa/vigqa_test_annotations.json",
    ]

    q2i, a2i, i2a, a2i_count = pre_process_dataset(image_dirs, qjsons, ajsons, img_prefix)
    print(list(map(len, (q2i, a2i, i2a, a2i_count))))

    with open('./outputs/coatt/q2i.pkl', 'wb') as f:
        pickle.dump(dict(q2i), f)
    with open('./outputs/coatt/a2i.pkl', 'wb') as f:
        pickle.dump(dict(a2i), f)
    with open('./outputs/coatt/i2a.pkl', 'wb') as f:
        pickle.dump(i2a, f)
    with open('./outputs/coatt/a2i_count.pkl', 'wb') as f:
        pickle.dump(a2i_count, f)

    np.save('./outputs/coatt/q2i.npy', q2i)
    np.save('./outputs/coatt/a2i.npy', a2i)
    np.save('./outputs/coatt/i2a.npy', i2a)
    np.save('./outputs/coatt/a2i_count.npy', a2i_count)

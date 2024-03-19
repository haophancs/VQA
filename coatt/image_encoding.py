import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from six.moves import cPickle as pickle

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class VqaImgDataset(Dataset):

    def __init__(self, image_dir, name, img_prefix):
        self.image_dir = image_dir
        self.img_names = [f for f in os.listdir(self.image_dir) if '.png' in f]
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor()])

        img_ids = {}
        for idx, fname in enumerate(self.img_names):
            img_id = fname.split('.')[0].split('_')[-1]
            img_ids[int(img_id)] = idx

        with open('./outputs/coatt/' + name + '_enc_idx.npy', 'wb') as f:
            pickle.dump(img_ids, f)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = default_loader(self.image_dir + '/' + self.img_names[idx])
        imgT = self.transform(img)

        return imgT.float()

tr_image_dir = './datasets/vigqa/train'
va_image_dir = './datasets/vigqa/val'
ts_image_dir = './datasets/vigqa/test'
tr_out_dir = './outputs/coatt/tr_enc'
va_out_dir = './outputs/coatt/va_enc'
ts_out_dir = './outputs/coatt/ts_enc'
DEVICE = 'cuda:0'

model = models.resnext50_32x4d(pretrained=True)
modules = list(model.children())[:-2]
model = nn.Sequential(*modules)
for params in model.parameters():
    params.requires_grad = False

if DEVICE == 'cuda:0':
    model = model.to('cuda:0')

tr_img_dataset = VqaImgDataset(image_dir=tr_image_dir, name='train', img_prefix="")
tr_img_dataset_loader = DataLoader(tr_img_dataset, batch_size=16, shuffle=False, num_workers=0)

va_img_dataset = VqaImgDataset(image_dir=va_image_dir, name='val', img_prefix="")
va_img_dataset_loader = DataLoader(va_img_dataset, batch_size=16, shuffle=False, num_workers=0)

ts_img_dataset = VqaImgDataset(image_dir=ts_image_dir, name='test', img_prefix="")
ts_img_dataset_loader = DataLoader(ts_img_dataset, batch_size=16, shuffle=False, num_workers=0)

print('Dumping Training images encodings.')
for idx, imgT in enumerate(tr_img_dataset_loader):
    imgT = imgT.to(DEVICE)
    out = model(imgT)
    out = out.view(out.size(0), out.size(1), -1)
    out = out.cpu().numpy()

    path = tr_out_dir + '/' + str(idx) + '.npz'
    #np.savez(path, out=out)
    np.savez_compressed(path, out=out)
    print(path)

print('Dumping Validation images encodings.')
for idx, imgT in enumerate(va_img_dataset_loader):
    imgT = imgT.to(DEVICE)
    out = model(imgT)
    out = out.view(out.size(0), out.size(1), -1)
    out = out.cpu().numpy()

    path = va_out_dir + '/' + str(idx) + '.npz'
    #np.savez(path, out=out)
    np.savez_compressed(path, out=out)
    print(path)

print('Dumping Validation images encodings.')
for idx, imgT in enumerate(ts_img_dataset_loader):
    imgT = imgT.to(DEVICE)
    out = model(imgT)
    out = out.view(out.size(0), out.size(1), -1)
    out = out.cpu().numpy()

    path = ts_out_dir + '/' + str(idx) + '.npz'
    #np.savez(path, out=out)
    np.savez_compressed(path, out=out)
    print(path)

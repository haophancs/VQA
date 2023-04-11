import os
import shutil
import json
import pickle
import nltk
from nltk.corpus import wordnet as wn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
from datetime import datetime
from torchvision import models
from googletrans import Translator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def wu_palmer_similarity(phrase1, phrase2):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens1 = [w.lower() for w in nltk.word_tokenize(phrase1) if w.lower() not in stopwords]
    tokens2 = [w.lower() for w in nltk.word_tokenize(phrase2) if w.lower() not in stopwords]

    synsets1 = set(wn.synsets(" ".join(tokens1)))
    synsets2 = set(wn.synsets(" ".join(tokens2)))
    if len(synsets1) == 0 or len(synsets2) == 0:
        return 0.0
    else:
        all_sim = []
        for synset1 in synsets1:
            for synset2 in synsets2:
                sim = synset1.wup_similarity(synset2)
                all_sim.append(sim)
        return np.mean(all_sim)

def create_trans_dict(words, translator):
    return dict(zip(
        words,
        list(map(
            lambda s: str.strip(s.lower()),
            translator.translate('\n'.join(words)).text.split('\n')
        ))
    ))


class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10, lr=0.001):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 250 * 4  # Steps
        self._save_freq = 1  # 2          # Epochs
        self._print_freq = 16
        self._batch_size = batch_size
        self._lr = lr

        # Use the GPU if it's available.
        self.DEVICE = "cuda"
        if self.DEVICE == "cuda":
            self._model = self._model.cuda()

        if self.method == 'simple':
            # self.optimizer = optim.Adam(self._model.parameters(), lr=self._lr)
            # self.optimizer = optim.SGD([{'params': self._model.embed.parameters(), 'lr': 0.8},
            #                            {'params': self._model.gnet.parameters(), 'lr': 1e-2},
            #                            {'params': self._model.fc.parameters(), 'lr': 1e-2}
            #                           ], momentum=0.9)
            self.optimizer = optim.Adam([{'params': self._model.embed.parameters(), 'lr': 0.08},
                                         {'params': self._model.gnet.parameters(), 'lr': 1e-3},
                                         {'params': self._model.fc.parameters(), 'lr': 1e-3}
                                         ], weight_decay=1e-8)
        else:
            self.optimizer = optim.Adam(self._model.parameters(), lr=self._lr, weight_decay=1e-8)
        self.criterion = nn.CrossEntropyLoss()
        self.initialize_weights()

        # Logger for tensorboard
        self.writer = SummaryWriter()

        if self.method == 'simple':
            self.chk_dir = './outputs/coatt/chk_simple/'
        else:
            self.chk_dir = './outputs/coatt/chk_coattention/'
            print('Creating Image Encoder')
            self.img_enc = models.resnet18(pretrained=True)
            modules = list(self.img_enc.children())[:-2]
            self.img_enc = nn.Sequential(*modules)
            for params in self.img_enc.parameters():
                params.requires_grad = False
            if self.DEVICE == "cuda":
                self.img_enc = self.img_enc.cuda()
            self.img_enc.eval()

        if not os.path.exists(self.chk_dir):
            os.makedirs(self.chk_dir)

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self):
        # TODO. Should return your validation accuracy
        all_pa = []
        all_ga = []
        for batch_id, (imgT, quesT, gT) in enumerate(self._val_dataset_loader):
            self._model.eval()  # Set the model to train mode

            if not self.method == 'simple':
                quesT = rnn.pack_sequence(quesT)
                imgT = imgT.to(self.DEVICE)
                imgT = self.img_enc(imgT)
                imgT = imgT.view(imgT.size(0), imgT.size(1), -1)

            imgT, quesT, gT = imgT.to(self.DEVICE), quesT.to(self.DEVICE), gT.to(self.DEVICE)
            gT = torch.squeeze(gT)
            pd_ans = self._model(imgT, quesT)  # TODO

            batch_pa = [torch.argmax(pd_ans[i]).item() for i in range(gT.shape[0])]
            batch_ga = gT.detach().cpu().numpy().tolist()
            all_pa += batch_pa
            all_ga += batch_ga

        return accuracy_score(all_ga, all_pa)

    def test(self):
        translator = Translator()
        # TODO. Should return your validation accuracy
        all_pa = []
        all_ga = []
        all_qid = []
        for batch_id, (imgT, quesT, gT) in enumerate(self._test_dataset_loader):
            all_qid += quesT

            self._model.eval()  # Set the model to train mode
            if not self.method == 'simple':
                quesT = rnn.pack_sequence(quesT)
                imgT = imgT.to(self.DEVICE)
                imgT = self.img_enc(imgT)
                imgT = imgT.view(imgT.size(0), imgT.size(1), -1)

            imgT, quesT, gT = imgT.to(self.DEVICE), quesT.to(self.DEVICE), gT.to(self.DEVICE)
            gT = torch.squeeze(gT)
            pd_ans = self._model(imgT, quesT)  # TODO

            batch_pa = [torch.argmax(pd_ans[i]).item() for i in range(gT.shape[0])]
            batch_ga = gT.detach().cpu().numpy().tolist()
            all_pa += batch_pa
            all_ga += batch_ga

        print("Test accuracy:", accuracy_score(all_ga, all_pa))
        print("Test macro precision:", precision_score(all_ga, all_pa, average='macro', zero_division=0))
        print("Test weighted precision:", precision_score(all_ga, all_pa, average='weighted', zero_division=0))
        print("Test macro recall:", recall_score(all_ga, all_pa, average='macro', zero_division=0))
        print("Test weighted recall:", recall_score(all_ga, all_pa, average='weighted', zero_division=0))
        print("Test macro f1:", f1_score(all_ga, all_pa, average='macro', zero_division=0))
        print("Test weighted f1:", f1_score(all_ga, all_pa, average='weighted', zero_division=0))

        with open('./outputs/coatt/i2a.pkl', 'rb') as f:
            i2a = pickle.load(f)
        all_ga = list(map(lambda a: i2a[a], all_ga))
        all_pa = list(map(lambda a: i2a[a], all_pa))

        pd.DataFrame.from_dict({
            'ga': all_ga, 'pa': all_pa
        }).to_csv('./outputs/.csv', index=False)

        trans_dict = create_trans_dict(np.union1d(all_ga, all_pa), translator=translator)
        wups = [wu_palmer_similarity(trans_dict[ga], trans_dict[pa]) for ga, pa in list(zip(all_ga, all_pa))]
        print("Test wups 0.0:", np.mean([it > 0.0 for it in wups]))
        print("Test wups 0.9:", np.mean([it > 0.9 for it in wups]))

    def train(self):
        print('Started Training.\n')
        tr_iter = 0
        val_iter = 0
        best_prec = 0.0
        for epoch in range(self._num_epochs):
            if (epoch + 1) // 3 == 0:
                self.adjust_learning_rate(epoch + 1)
            num_batches = len(self._train_dataset_loader)

            for batch_id, (imgT, quesT, gT) in enumerate(self._train_dataset_loader):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                # ============
                # TODO: Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                if not self.method == 'simple':
                    quesT = rnn.pack_sequence(quesT)
                    imgT = imgT.to(self.DEVICE)
                    imgT = self.img_enc(imgT)
                    imgT = imgT.view(imgT.size(0), imgT.size(1), -1)
                else:
                    imgT = imgT.to(self.DEVICE)

                quesT, gT = quesT.to(self.DEVICE), gT.to(self.DEVICE)
                predicted_answer = self._model(imgT, quesT)  # TODO
                ground_truth_answer = torch.squeeze(gT)  # TODO
                # ============

                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, ground_truth_answer)

                if (current_step + 1) % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))

                    # TODO: you probably want to plot something here
                    self.writer.add_scalar('train/loss', loss.item(), tr_iter)
                    tr_iter = tr_iter + 1

            #                if (current_step + 1) % self._test_freq == 0:
            #                    self._model.eval()
            #                    val_accuracy = self.validate()
            #                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
            #
            #                    # TODO: you probably want to plot something here
            #                    self.writer.add_scalar('valid/accuracy', val_accuracy, val_iter)
            #                    val_iter = val_iter + 1

            if (epoch + 1) % self._save_freq == 0 or epoch == self._num_epochs - 1:
                val_accuracy = self.validate()
                print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                self.writer.add_scalar('valid/accuracy', val_accuracy, val_iter)
                val_iter = val_iter + 1

                # remember best val_accuracy and save checkpoint
                is_best = val_accuracy > best_prec
                best_prec = max(val_accuracy, best_prec)
                self.save_checkpoint({'epoch': epoch + 1,
                                      'state_dict': self._model.state_dict(),
                                      'best_prec': best_prec},
                                     # 'optimizer': optimizer.state_dict()}, is_best,
                                     is_best, self.chk_dir + 'checkpoint_' + str(epoch + 1) + '.pth.tar')
            self.test()

        # Closing tensorboard logger
        logdir = os.path.join('./outputs/coatt/log/tb_', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.writer.export_scalars_to_json(logdir + 'tb_summary.json')
        self.writer.close()

    def initialize_weights(self):
        for layer in self._model.modules():
            if not isinstance(layer, (nn.Conv2d, nn.Linear)):
                continue
            try:
                torch.nn.init.xavier_normal_(layer.weight)
                try:
                    nn.init.constant_(layer.bias.data, 0)
                except:
                    pass
            except:
                pass

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10

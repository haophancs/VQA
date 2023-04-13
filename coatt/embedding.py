import os
from abc import abstractmethod

import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class PretrainedTextEmbedding:
    def __init__(self, embed_dim=None, device='cpu', **kwargs):
        self.embed_dim = embed_dim
        self.embed_layer = None
        self.device = torch.device(device)

    @staticmethod
    def from_pretrained(pretrained_name, embed_dim=None, pretrained_root=None, device='cpu'):
        print(f'Load pretrained {pretrained_name} for text embedding')
        if not pretrained_root:
            pretrained_root = ''
        if pretrained_name in PretrainedSeq2SeqEmbedding.PRETRAINED_LIST:
            return PretrainedSeq2SeqEmbedding(
                embed_dim=embed_dim,
                pretrained_name=pretrained_name,
                device=device
            )
        if pretrained_name in PretrainedW2VEmbedding.PRETRAINED_LIST:
            return PretrainedW2VEmbedding(
                pretrained_name=pretrained_name,
                pretrained_root=pretrained_root,
                embed_dim=embed_dim,
                device=device
            )

    @abstractmethod
    def __call__(self, textual_tokens, **kwargs):
        raise NotImplementedError()

    def _reshape_to_embed_dim(self, embeddings):
        if self.embed_dim:
            if self.embed_dim > embeddings.shape[2]:
                embeddings = torch.cat([
                    embeddings,
                    torch.zeros(size=(
                        embeddings.shape[0],
                        embeddings.shape[1],
                        self.embed_dim - embeddings.shape[2]
                    )).to(self.device)
                ], dim=-1)
            else:
                embeddings = embeddings[:, :, :self.embed_dim]
        return embeddings


class PretrainedW2VEmbedding(PretrainedTextEmbedding):
    _NAME_TO_FILE = {
        'phow2v.syllable.100d': 'word2vec_vi_syllables_100dims.txt',
        'phow2v.syllable.300d': 'word2vec_vi_syllables_300dims.txt',
        'phow2v.word.100d': 'word2vec_vi_words_100dims.txt',
        'phow2v.word.300d': 'word2vec_vi_words_300dims.txt'
    }
    PRETRAINED_LIST = list(_NAME_TO_FILE.keys())

    def __init__(self, pretrained_name, pretrained_root, embed_dim=None, device='cpu', **kwargs):
        super().__init__(embed_dim=embed_dim, device=device, **kwargs)
        self.embed_layer = nn.Embedding.from_pretrained(
            torch.FloatTensor(gensim.models.KeyedVectors.load_word2vec_format(os.path.join(
                pretrained_root,
                PretrainedW2VEmbedding._NAME_TO_FILE[pretrained_name]
            )).vectors)).to(self.device)
        self.embed_layer.requires_grad = False

    def __call__(self, textual_tokens, **kwargs):
        embeddings = self.embed_layer(textual_tokens.detach().to(self.device))
        embeddings = self._reshape_to_embed_dim(embeddings).to(textual_tokens.device)
        return embeddings


class PretrainedSeq2SeqEmbedding(PretrainedTextEmbedding):
    PRETRAINED_LIST = ["vinai/bartpho-syllable", "vinai/bartpho-word"]

    def __init__(
            self,
            pretrained_name,
            pretrained_root='huggingface',
            embed_dim=None,
            device='cpu',
            **kwargs
    ):
        super().__init__(embed_dim=embed_dim, device=device, **kwargs)
        self.embed_layer = AutoModel.from_pretrained(pretrained_name).to(self.device)
        self.embed_layer.requires_grad = False

    def __call__(self, textual_tokens, attention_mask=None, **kwargs):
        embeddings = self.embed_layer.forward(
            input_ids=textual_tokens.detach().to(self.device),
            attention_mask=attention_mask.detach().to(self.device) if attention_mask else None
        ).last_hidden_state
        embeddings = self._reshape_to_embed_dim(embeddings).to(textual_tokens.device)
        return embeddings
'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
# !pip install sentencepiece

data_dir = "/content"

! pip list | grep sentencepiece

import sentencepiece as spm
import csv
import sys
import os
import math
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unicodedata

from tqdm import tqdm, tqdm_notebook, trange

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from IPython.display import display

# Setup seeds
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# for using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
D3. [PASS] Tokenizer Install & import
'''
# Keras Tokenizer is a tokenizer provided by default in tensorflow 2.X and is a word level tokenizer. It does not require a separate installation.

'''
D4. Define Hyperparameters for Data Engineering
'''
ENCODER_LEN  = 15
DECODER_LEN  = 23
BATCH_SIZE   = 16

'''
D5. Load and modifiy to pandas dataframe
'''
import pandas as pd

pd.set_option('display.max_colwidth', None)

"""
raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ("A man's worth lies in what he is.", "La valeur d'un homme réside dans ce qu'il est."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("All three of you need to do that.", "Vous avez besoin de faire cela, tous les trois."),
    ("Are you giving me another chance?", "Me donnez-vous une autre chance ?"),
    ("Both Tom and Mary work as models.", "Tom et Mary travaillent tous les deux comme mannequins."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"),

    ("Could you close the door, please?", "Pourriez-vous fermer la porte, s'il vous plaît ?"),
    ("Did you plant pumpkins this year?", "Cette année, avez-vous planté des citrouilles ?"),
    ("Do you ever study in the library?", "Est-ce que vous étudiez à la bibliothèque des fois ?"),
    ("Don't be deceived by appearances.", "Ne vous laissez pas abuser par les apparences."),
    ("Excuse me. Can you speak English?", "Je vous prie de m'excuser ! Savez-vous parler anglais ?"),
    ("Few people know the true meaning.", "Peu de gens savent ce que cela veut réellement dire."),
    ("Germany produced many scientists.", "L'Allemagne a produit beaucoup de scientifiques."),
    ("Guess whose birthday it is today.", "Devine de qui c'est l'anniversaire, aujourd'hui !"),

    ("He acted like he owned the place.", "Il s'est comporté comme s'il possédait l'endroit."),
    ("Honesty will pay in the long run.", "L'honnêteté paye à la longue."),
    ("How do we know this isn't a trap?", "Comment savez-vous qu'il ne s'agit pas d'un piège ?"),
    ("I can't believe you're giving up.", "Je n'arrive pas à croire que vous abandonniez."),
    ("I have something very important to tell you.", "Il me faut vous dire quelque chose de très important."),
    ("I have three times as many books as he does.", "J'ai trois fois plus de livres que lui."),
    ("I have to change the batteries in the radio.", "Il faut que je change les piles de cette radio."),
    ("I have to finish up some things before I go.", "Je dois finir deux trois trucs avant d'y aller."),

    ("I have to think about what needs to be done.", "Je dois réfléchir sur ce qu'il faut faire."),
    ("I haven't been back here since the incident.", "Je ne suis pas revenu ici depuis l'accident."),
    ("I haven't eaten anything since this morning.", "Je n'ai rien mangé depuis ce matin."),
    ("I hear his business is on the verge of ruin.", "Apparemment son entreprise est au bord de la faillite."),
    ("I hope I didn't make you feel uncomfortable.", "J'espère que je ne t'ai pas mis mal à l'aise."),
    ("I hope to continue to see more of the world.", "J'espère continuer à voir davantage le monde."),
    ("I hope to see reindeer on my trip to Sweden.", "J'espère voir des rennes lors de mon voyage en Suède."),
    ("I hope you'll find this office satisfactory.", "J'espère que ce bureau vous conviendra."),

    ("I hurried in order to catch the first train.", "Je me dépêchai pour avoir le premier train."),
    ("I just can't stand this hot weather anymore.", "Je ne peux juste plus supporter cette chaleur."),
    ("I just don't want there to be any bloodshed.", "Je ne veux tout simplement pas qu'il y ait une effusion de sang."),
    ("I just thought that you wouldn't want to go.", "J'ai simplement pensé que vous ne voudriez pas y aller."),
    ("I plan to go. I don't care if you do or not.", "Je prévois d'y aller. Ça m'est égal que vous y alliez aussi ou pas."),
    ("I prefer soap as a liquid rather than a bar.", "Je préfère le savon liquide à une savonnette."),
    ("I promise you I'll explain everything later.", "Je vous promets que j'expliquerai tout plus tard."),
    ("I ran as fast as I could to catch the train.", "Je courus aussi vite que je pus pour attraper le train."))


raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"))
"""

raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ("A man's worth lies in what he is.", "La valeur d'un homme réside dans ce qu'il est."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("All three of you need to do that.", "Vous avez besoin de faire cela, tous les trois."),
    ("Are you giving me another chance?", "Me donnez-vous une autre chance ?"),
    ("Both Tom and Mary work as models.", "Tom et Mary travaillent tous les deux comme mannequins."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"),
    
    ("Could you close the door, please?", "Pourriez-vous fermer la porte, s'il vous plaît ?"),
    ("Did you plant pumpkins this year?", "Cette année, avez-vous planté des citrouilles ?"),
    ("Do you ever study in the library?", "Est-ce que vous étudiez à la bibliothèque des fois ?"),
    ("Don't be deceived by appearances.", "Ne vous laissez pas abuser par les apparences."),
    ("Excuse me. Can you speak English?", "Je vous prie de m'excuser ! Savez-vous parler anglais ?"),
    ("Few people know the true meaning.", "Peu de gens savent ce que cela veut réellement dire."),
    ("Germany produced many scientists.", "L'Allemagne a produit beaucoup de scientifiques."),
    ("Guess whose birthday it is today.", "Devine de qui c'est l'anniversaire, aujourd'hui !"))

import unicodedata
import re

from tensorflow.keras.preprocessing.text import Tokenizer

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
    
def preprocess(sent):
    # 위에서 구현한 함수를 내부적으로 호출
    sent = unicode_to_ascii(sent.lower())

    # 단어와 구두점 사이에 공백을 만듭니다.
    # Ex) "he is a boy." => "he is a boy ."
    sent = re.sub(r"([?.!,¿])", r" \1", sent)

    # (a-z, A-Z, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환합니다.
    sent = re.sub(r"[^a-zA-Z!.?]+", r" ", sent)

    sent = re.sub(r"\s+", " ", sent)
    return sent

# 인코딩 테스트
en_sent = u"Have you had dinner?"
fr_sent = u"Avez-vous deja dine?"

print(preprocess(en_sent))
print(preprocess(fr_sent).encode('utf-8'))

raw_encoder_input, raw_data_fr = list(zip(*raw_data))
raw_encoder_input, raw_data_fr = list(raw_encoder_input), list(raw_data_fr)

raw_src = [preprocess(data) for data in raw_encoder_input]
raw_trg = [preprocess(data) for data in raw_data_fr]

print(raw_src[:4])
print(raw_trg[:4])

'''
D9. Define dataframe
'''
SRC_df = pd.DataFrame(raw_src)
TRG_df = pd.DataFrame(raw_trg)

SRC_df.rename(columns={0: "SRC"}, errors="raise", inplace=True)
TRG_df.rename(columns={0: "TRG"}, errors="raise", inplace=True)
train_df = pd.concat([SRC_df, TRG_df], axis=1)

print('Translation Pair :',len(train_df)) # 리뷰 개수 출력
train_df.sample(3)

raw_src_df  = train_df['SRC']
raw_trg_df  = train_df['TRG']

src_sentence  = raw_src_df
trg_sentence  = raw_trg_df

'''
D10. Define tokenizer
'''

with open('corpus_src.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(train_df['SRC']))

with open('corpus_trg.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(train_df['TRG']))

# This is the folder to save the data. Modify it to suit your environment.
data_dir = "/content"

corpus = "corpus_src.txt"
prefix = "nmt_src_vocab"
vocab_size = 200
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" +               # max sentence length
    " --pad_id=0 --pad_piece=[PAD]" +               # pad (0)
    " --unk_id=1 --unk_piece=[UNK]" +               # unknown (1)
    " --bos_id=2 --bos_piece=[BOS]" +               # begin of sequence (2)
    " --eos_id=3 --eos_piece=[EOS]" +               # end of sequence (3)
    " --user_defined_symbols=[SEP],[CLS],[MASK]")   # other additional tokens

corpus = "corpus_trg.txt"
prefix = "nmt_trg_vocab"

vocab_size = 200
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" +               # max sentence length
    " --pad_id=0 --pad_piece=[PAD]" +               # pad (0)
    " --unk_id=1 --unk_piece=[UNK]" +               # unknown (1)
    " --bos_id=2 --bos_piece=[BOS]" +               # begin of sequence (2)
    " --eos_id=3 --eos_piece=[EOS]" +               # end of sequence (3)
    " --user_defined_symbols=[SEP],[CLS],[MASK]")   # other additional tokens

for f in os.listdir("."):
    print(f)

vocab_src_file = f"{data_dir}/nmt_src_vocab.model"
vocab_src = spm.SentencePieceProcessor()
vocab_src.load(vocab_src_file)

vocab_trg_file = f"{data_dir}/nmt_trg_vocab.model"
vocab_trg = spm.SentencePieceProcessor()
vocab_trg.load(vocab_trg_file)

n_enc_vocab = len(vocab_src)
n_dec_vocab = len(vocab_trg)

print('Word set size of Encoder :',n_enc_vocab)
print('Word set size of Decoder :',n_dec_vocab)

'''
Token List
'''
# Recommend : For small number of vocabulary, please test each IDs.
# src_vocab_list
src_vocab_list = [[vocab_src.id_to_piece(id), id] for id in range(vocab_src.get_piece_size())]

# trg_vocab_list
trg_vocab_list = [[vocab_trg.id_to_piece(id), id] for id in range(vocab_trg.get_piece_size())]

'''
D11. Tokenizer test
'''
# Source Tokenizer
lines = [  SRC_df.iloc[1,0],  SRC_df.iloc[2,0],  SRC_df.iloc[3,0]]
for line in lines:
    print("Input        :", line)
    txt_2_ids = vocab_src.encode_as_ids(line)
    print("EncodeIds    :", txt_2_ids)
    print("DecodeIds    :", vocab_src.DecodeIds(txt_2_ids))

    txt_2_tkn = vocab_src.encode_as_pieces(line)
    print("EncodePieces :", txt_2_tkn)
    print("DecodePieces :", vocab_src.DecodePieces(txt_2_tkn))

    ids2 = vocab_src.piece_to_id(txt_2_tkn)
    print("Piece_2_IDs  :", ids2)
    print("Id_2_Pieces  :", vocab_src.id_to_piece(ids2))
    print("\n")

print("\n")

# Target Tokenizer
lines = [  TRG_df.iloc[1,0],  TRG_df.iloc[2,0],  TRG_df.iloc[3,0]]
for line in lines:
    print("Input        :", line)
    txt_2_ids = vocab_trg.encode_as_ids(line)
    print("EncodeIds    :", txt_2_ids)
    print("DecodeIds    :", vocab_trg.DecodeIds(txt_2_ids))
    
    txt_2_tkn = vocab_trg.encode_as_pieces(line)
    print("EncodePieces :", txt_2_tkn)
    print("DecodePieces :", vocab_trg.DecodePieces(txt_2_tkn))

    ids2 = vocab_trg.piece_to_id(txt_2_tkn)
    print("Piece_2_IDs  :", ids2)
    print("Id_2_Pieces  :", vocab_trg.id_to_piece(ids2))
    print("\n")

'''
D12. Tokenize
'''
# tokenize / encode integers / add start and end tokens / padding
tokenized_src  = vocab_src.encode_as_ids(src_sentence.to_list())
tokenized_trg  = vocab_trg.encode_as_ids(trg_sentence.to_list())

# Add [BOS], [EOS] token ids to each target list elements.
new_list = [ x.insert(0, 2) for x in tokenized_trg]
new_list = [ x.insert(len(x), 3) for x in tokenized_trg]

tokenized_inputs  = tokenized_src
tokenized_outputs = tokenized_trg

'''
D13. [EDA] Explore the tokenized datasets
'''

len_result = [len(s) for s in tokenized_inputs]

print('Maximum length of source : {}'.format(np.max(len_result)))
print('Average length of source : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()

len_result = [len(s) for s in tokenized_outputs]

print('Maximum length of target : {}'.format(np.max(len_result)))
print('Average length of target : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()

'''
D14. Pad sequences
'''

from tensorflow.keras.preprocessing.sequence import pad_sequences
tkn_sources = pad_sequences(tokenized_inputs,  maxlen=ENCODER_LEN, padding='post', truncating='post')
tkn_targets = pad_sequences(tokenized_outputs, maxlen=DECODER_LEN, padding='post', truncating='post')

'''
D15. Send data to device
'''

tensors_src   = torch.tensor(tkn_sources).to(device)
tensors_trg   = torch.tensor(tkn_targets).to(device)

'''
D16. [EDA] Explore the Tokenized datasets
'''
print('Size of source language data(shape) :', tkn_sources.shape)
print('Size of target language data(shape) :', tkn_targets.shape)

# Randomly output the 0th sample
print(tkn_sources[0])
print(tkn_targets[0])

'''
D17. [PASS] Split Data
'''

'''
D18. Build dataset
'''

from torch.utils.data import TensorDataset   # 텐서데이터셋
from torch.utils.data import DataLoader      # 데이터로더

dataset    = TensorDataset(tensors_src, tensors_trg)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


'''
D19. [PASS] Define some useful parameters for further use
'''

'''
Model Engineering
'''

'''
M01. Import Libraries for Model Engineering
'''
from tqdm import tqdm, tqdm_notebook, trange

! pip install hyper_connections 

from collections import namedtuple
from functools import wraps
from packaging import version

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# constants

Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def exists(val):
    return val is not None

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# main class

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        causal = False,
        use_flash = False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.use_flash = use_flash
        assert not (use_flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask):
            if mask.ndim != 4:
                mask = rearrange(mask, 'b j -> b 1 1 j')

            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = self.causal
            )

        return out

    def forward(self, q, k, v, mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device = q.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q, k, v, mask = mask)

        # similarity

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        # key padding mask

        if exists(mask):
            if mask.ndim != 4:
                mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask

        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out

from __future__ import annotations

import math
from functools import partial
from itertools import zip_longest
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# from recurrent_memory_transformer_pytorch.attend import Attend

from hyper_connections import get_init_and_expand_reduce_stream_functions

# constants

Linear = partial(nn.Linear, bias = False)

# helpers

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

def divisible_by(numer, denom):
    return (numer % denom) == 0

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def frac_gradient(t, frac = 1.):
    if frac == 1.:
        return t

    return t * frac + t.detach() * (1. - frac)

# rotary embedding

class RotaryEmbedding(Module):
    def __init__(self, dim, theta = 32768):
        super().__init__()
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, positions):
        freqs = torch.einsum('i , j -> i j', positions, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

# feedforward

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_inner = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.RMSNorm(dim),
        Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        Linear(dim_inner, dim)
    )

# attention

class Attention(Module):
    def __init__(
        self,
        *,
        dim,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        accept_value_residual = False,
        use_flash_attn = False,
        use_custom_causal_attn_mask = False
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)

        dim_inner = dim_head * heads
        self.heads = heads

        self.attend = Attend(
            causal = causal and not use_custom_causal_attn_mask,
            dropout = dropout,
            use_flash = use_flash_attn
        )

        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))

        self.to_q = Linear(dim, dim_inner)
        self.to_kv = Linear(dim, dim_inner * 2)
        self.to_out = Linear(dim_inner, dim)

        # learned value residual mixing

        self.learned_value_residual_mix = None

        if accept_value_residual:
            self.learned_value_residual_mix = nn.Sequential(
                Linear(dim, heads),
                Rearrange('b n h -> b h n 1'),
                nn.Sigmoid()
            )

    def forward(
        self,
        x,
        rotary_emb: tuple[Tensor, Tensor] | None = None,
        mask = None,
        xl_memories = None,
        value_residual = None
    ):
        assert not (exists(value_residual) ^ exists(self.learned_value_residual_mix))

        h = self.heads
        x = self.norm(x)

        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # handle value residual

        orig_v = v

        if exists(self.learned_value_residual_mix):
            mix = self.learned_value_residual_mix(x)
            v = v.lerp(value_residual, mix)

        # add a null key / value
        # to protect against an entirely masked out sequence
        # as well as giving attention ability to attend to nothing

        nk, nv = map(lambda t: repeat(t, 'h d -> b h 1 d', b = x.shape[0]), self.null_kv)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)

        # manage memories

        next_xl_memories = torch.stack((k, v))

        if exists(xl_memories):
            kx, vx = xl_memories
            k = torch.cat((kx, k), dim = -2)
            v = torch.cat((vx, v), dim = -2)

            if exists(mask):
                mask = F.pad(mask, (xl_memories.shape[-2], 0), value = True)

        if exists(rotary_emb):
            q_rotary_emb, k_rotary_emb = rotary_emb

            q = apply_rotary_pos_emb(q_rotary_emb, q)
            k = apply_rotary_pos_emb(k_rotary_emb, k)

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), next_xl_memories, orig_v

# transformer

class RecurrentMemoryTransformer(Module):
    def __init__(
        self,
        dim,
        *,
        num_tokens,
        depth,
        num_memory_tokens,
        seq_len,
        causal = True,        
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        use_flash_attn = False,
        ignore_index = -1,
        abs_pos_emb = True,
        rotary_pos_emb = False,
        use_xl_memories = True,
        xl_mem_len = None,
        enhanced_xl_recurrence = False,      # add simple method for enhancing receptive field of xl memories, from ernie-doc paper
        emb_gradient_frac = 0.1,             # trick from cogview paper that leads to a bit more stability
        memory_not_causal = True,            # flash attention behaves a bit more optimally if causal mask is not explicitly passed in - but if the memories perform better without a causal mask, it is necessary to have this turned on
        add_write_to_next_write_mem = False, # add the write memories of previous step to the next write step - thanks to @IcarusWizard for pointing out this discrepancy
        next_write_mem_stop_grad = True,     # whether to stop gradient of previous read memory -> next write memory
        always_have_read_memories = True,    # whether to always have read memories, even on the first step, so to make the model onnx-able
        num_residual_streams = 4             # number of residual streams for hyper connections
    ):
        super().__init__()
        self.causal = causal
        self.seq_len = seq_len

        self.emb_gradient_frac = emb_gradient_frac

        assert num_memory_tokens > 0

        self.token_emb = nn.Embedding(num_tokens, dim)

        # positions

        assert any([abs_pos_emb, rotary_pos_emb])

        self.pos_emb = nn.Embedding(seq_len, dim) if abs_pos_emb else None

        self.rotary_pos_emb = RotaryEmbedding(dim_head) if rotary_pos_emb else None

        # memory related

        self.num_memory_tokens = num_memory_tokens

        self.read_memory_emb = nn.Parameter(torch.zeros(num_memory_tokens, dim))
        nn.init.normal_(self.read_memory_emb, std = 0.02)

        self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))
        nn.init.normal_(self.memory_tokens, std = 0.02)

        # xl memories

        xl_mem_len = default(xl_mem_len, seq_len)
        assert xl_mem_len <= seq_len
        self.xl_mem_len = xl_mem_len

        self.use_xl_memories = use_xl_memories
        self.enhanced_xl_recurrence = enhanced_xl_recurrence

        # hyper connections

        init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        # layers

        self.layers = ModuleList([])

        for layer_index in range(depth):
            is_first = layer_index == 0

            self.layers.append(ModuleList([
                init_hyper_conn(dim = dim, branch = Attention(
                    dim = dim,
                    dim_head = dim_head,
                    causal = causal,
                    heads = heads,
                    use_flash_attn = use_flash_attn,
                    accept_value_residual = not is_first,
                    use_custom_causal_attn_mask = memory_not_causal,
                    dropout = attn_dropout
                )),
                init_hyper_conn(dim = dim, branch = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)),
            ]))

        self.norm = nn.RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens)

        self.ignore_index = ignore_index

        # whether to use custom attention mask if causal and memory should not be causal

        self.use_custom_causal_attn_mask = causal and memory_not_causal

        # in the paper, they actually also use the previous write memories for the next write memories

        self.add_write_to_next_write_mem = add_write_to_next_write_mem
        self.next_write_mem_stop_grad = next_write_mem_stop_grad

        # allow for attending to raw read memory positional embeddings on first step
        # hack to make it onnx-able and should not hurt

        self.always_have_read_memories = always_have_read_memories

    def init_memory(self, batch):
        return repeat(self.memory_tokens, 'm d -> b m d', b = batch)

    def forward(
        self,
        x,
        read_memories = None,
        *,
        mask = None,
        labels = None,
        xl_memories: list[Tensor] | None = None,
        mask_out_read_memories = False   # in the case one is passing in 0s for read memories, for onnx-able model
    ):
        has_xl_memories = exists(xl_memories) and len(xl_memories) > 0

        b, n, device, mem_length, return_loss = *x.shape, x.device, self.num_memory_tokens, exists(labels)

        assert n <= self.seq_len

        pos = torch.arange(n, device = device)

        x = self.token_emb(x)

        # maybe absolute positional embedding

        if exists(self.pos_emb):
            x = x + self.pos_emb(pos)

        # trick from cogview paper

        x = frac_gradient(x, self.emb_gradient_frac)

        # prepare write memories, as in paper

        write_memories = self.init_memory(b)

        if exists(read_memories) and self.add_write_to_next_write_mem:
            maybe_detach = torch.detach if self.next_write_mem_stop_grad else identity
            write_memories = write_memories + maybe_detach(read_memories)

        # prepare read memories

        if exists(read_memories):
            if read_memories.ndim == 2:
                read_memories = repeat(read_memories, 'n d -> b n d', b = b)

            read_mem_length = mem_length
            read_memories = read_memories + self.read_memory_emb
        elif self.always_have_read_memories:
            read_mem_length = mem_length
            read_memories = repeat(self.read_memory_emb, 'n d -> b n d', b = b)
        else:
            read_mem_length = 0
            read_memories = x[:, 0:0]

        # concat to main sequence using einop's pack

        x, ps = pack([read_memories, x, write_memories], 'b * d')

        # take care of mask

        if exists(mask):
            mask = F.pad(mask, (read_mem_length, mem_length), value = True)

        # custom causal mask, if needed

        if self.use_custom_causal_attn_mask:
            causal_mask = torch.ones((n, n), device = device, dtype = torch.bool).tril()

            causal_mask = F.pad(causal_mask, (0, mem_length, read_mem_length, 0), value = False)
            causal_mask = F.pad(causal_mask, (read_mem_length, 0, 0, mem_length), value = True)

            causal_mask = rearrange(causal_mask, 'i j -> 1 1 i j')

            if exists(mask):
                mask = rearrange(mask, 'b j -> b 1 1 j')
                mask = mask & causal_mask
            else:
                mask = causal_mask

        # masking out read memories, either for passing in 0s for read memories on first step, or if you are doing some regularization game on the memories

        if read_mem_length > 0 and mask_out_read_memories:
            read_mem_mask = torch.arange(x.shape[-2], device = device) < read_mem_length

            if exists(mask):
                mask = mask & ~read_mem_mask
            else:
                mask = read_mem_mask

        # rotary embedding - offset main positions by 10000, and keep all memories at position 0

        rotary_emb = None

        if exists(self.rotary_pos_emb):
            mem_rel_dist = 10000

            q_pos = pos + mem_rel_dist

            if has_xl_memories:
                xl_mem_length = xl_memories[0].shape[-2]
                q_pos += xl_mem_length

            q_pos = F.pad(q_pos, (read_mem_length, mem_length), value = 0)
            q_rotary_emb = self.rotary_pos_emb(q_pos)

            # kind of confusing at the moment
            # but the order of the keys are - [xl memories] [read memories] [main sequence] [ write memories]
            # so the positions are (say xl memory length of 256) - [10001, 10002, 10003 ...] [0, 0, ...] [10256, 10257, ...] [0, 0, ...]

            if has_xl_memories:
                k_pos = torch.arange(xl_mem_length, device = device) + mem_rel_dist
                k_pos = torch.cat((k_pos, q_pos), dim = -1)
            else:
                k_pos = q_pos

            # account for null key / value

            k_pos = F.pad(k_pos, (1, 0), value = mem_rel_dist - 1) # give a null memory token, to allow for attending to nothing

            k_rotary_emb = self.rotary_pos_emb(k_pos)

            rotary_emb = (q_rotary_emb, k_rotary_emb)

        # prepare xl memories

        xl_memories = default(xl_memories, [])
        xl_memories_iter = iter(xl_memories)
        new_xl_memories = []

        if has_xl_memories and self.enhanced_xl_recurrence and len(xl_memories) > 1:  # simply shift all the xl memories down by one, so lower layer gets access to representations from layer above
            xl_memories = [*xl_memories[1:], xl_memories[0]]

        # value residual

        value_residual = None

        # expand streams for hyper connections

        x = self.expand_streams(x)

        # attention and feedforward

        for attn, ff in self.layers:
            x, xl_memories, attn_values = attn(x, mask = mask, xl_memories = next(xl_memories_iter, None), rotary_emb = rotary_emb, value_residual = value_residual)

            value_residual = default(value_residual, attn_values)
            new_xl_memories.append(xl_memories)

            x = ff(x)

        # reduce streams for hyper connections

        x = self.reduce_streams(x)

        # final norm

        x = self.norm(x)

        # whether to return xl memories

        next_xl_memories = None

        if self.use_xl_memories:
            next_xl_memories = list(map(lambda t: torch.detach(t[..., -self.xl_mem_len:, :]), new_xl_memories))

        # split out memories using unpack

        read_memories, x, write_memories = unpack(x, ps, 'b * d')

        # to logits

        logits = self.to_logits(x)

        if not return_loss:
            return logits, write_memories, next_xl_memories

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss, write_memories, next_xl_memories

# wrapper to manage many segments

class RecurrentMemoryTransformerWrapper(Module):
    def __init__(
        self,
        transformer: RecurrentMemoryTransformer,
        truncate_at_step = None  # number of steps before detaching memories (truncated bptt). with memory replay checkpointing, there should be no memory issues, but in case of instability, as reported in initial paper
    ):
        super().__init__()
        self.transformer = transformer
        self.seq_len = transformer.seq_len
        self.truncate_at_step = truncate_at_step

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prime,
        *,
        length,
        memories = None,
        xl_memories: list[Tensor] | None = None,
        temperature = 1.,
        filter_thres = 0.9
    ):
        assert self.transformer.causal, 'only autoregressive transformers can generate'

        start_len, seq_len = prime.shape[-1], self.seq_len

        assert length >= start_len

        *past_segments, curr_segment = prime.split(seq_len, dim = -1)

        # catch memories up to the current segment

        for past_segment in past_segments:
            _, memories, xl_memories = self.transformer(past_segment, memories, xl_memories = xl_memories)

        # sample for the remaining length

        for ind in range(length - start_len):
            logits, next_memories, next_xl_memories = self.transformer(curr_segment, memories, xl_memories = xl_memories)

            logits = logits[:, -1]

            filtered_logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature)
            sampled = rearrange(sampled, 'b -> b 1')

            curr_segment = torch.cat((curr_segment, sampled), dim = -1)

            if divisible_by(curr_segment.shape[-1] - 1, seq_len):
                memories = next_memories
                xl_memories = next_xl_memories

                past_segment, curr_segment = curr_segment[..., :seq_len], curr_segment[..., -1:]
                past_segments.append(past_segment)

        # add current segment to all segments

        past_segments.append(curr_segment)

        # reconcat all segments

        output = torch.cat(past_segments, dim = -1)

        output = output[:, start_len:]
        return output

    def forward(
        self,
        x,
        memories = None,
        *,
        mask = None,
        xl_memories: list[Tensor] | None = None,
        return_loss = False,
        labels = None,
        truncate_at_step = None,         # if set, this would override the truncate_at_step at init
        memory_replay_backprop = False,  # whether to have the class do the backwards pass memory efficiently
        mrbp_loss_weight = 1.            # if using memory replay backprop with gradient accumulation, scale loss by this factor ex. (1. / <num grad accum steps>)
    ):
        seq_len, truncate_at_step = self.seq_len, default(truncate_at_step, self.truncate_at_step)

        labels = None
        if (return_loss or memory_replay_backprop) and not exists(labels):
            x, labels = x[:, :-1], x[:, 1:]

        # segment input

        segments = x.split(seq_len, dim = -1)
        total_length = x.shape[-1]
        num_segments = len(segments)
        segment_length_frac = tuple(map(lambda t: t.shape[-1] / total_length, segments))

        # default values

        label_segments = mask_segments = (None,)

        # take care of labels

        if exists(labels):
            label_segments = labels.split(seq_len, dim = -1)

        # take care of the mask

        if exists(mask):
            mask_segments = mask.split(seq_len, dim = -1)

        # keep replay buffer

        replay_buffer = [memories]

        # replay buffer for xl memories

        xl_segments = [xl_memories]

        # decide context of forward depending on whether doing memory-replay-backprop

        forward_context = nullcontext if not memory_replay_backprop else torch.no_grad

        # forward and get all outputs (can be either loss or logits)

        logits = []
        losses = []

        for step, (segment, mask_segment, label_segment, loss_weight) in enumerate(zip_longest(segments, mask_segments, label_segments, segment_length_frac)):

            with forward_context():
                output, memories, xl_memories = self.transformer(segment, memories, mask = mask_segment, labels = label_segment)

            if exists(truncate_at_step) and divisible_by(step + 1, truncate_at_step):
                memories = memories.detach()

            replay_buffer.append(memories)

            xl_segments.append(xl_memories)

            if return_loss:
                losses.append(output * loss_weight)
            else:
                logits.append(output)

        # whether to do memory replay backpropagation

        # https://arxiv.org/abs/2010.06891
        # algorithm 1

        if memory_replay_backprop:
            memories_grad = torch.zeros_like(replay_buffer[-1])

            reversed_inputs = zip_longest(*map(reversed, [
                range(num_segments),
                segments,
                replay_buffer[:-1],
                xl_segments[:-1],
                mask_segments,
                label_segments,
                segment_length_frac,
            ]))

            total_loss = 0.

            for step, segment, segment_memories, segment_xl_memories, mask_segment, label_segment, loss_weight in reversed_inputs:
                is_first = step == 0

                if exists(segment_memories):
                    segment_memories.requires_grad_()

                loss, next_segment_memories, _ = self.transformer(segment, segment_memories, mask = mask_segment, xl_memories = segment_xl_memories, labels = label_segment)

                weighted_loss = loss * loss_weight * mrbp_loss_weight

                weighted_loss.backward(retain_graph = True)

                next_segment_memories.backward(memories_grad)

                total_loss += weighted_loss

                if is_first:
                    continue

                if exists(truncate_at_step) and divisible_by(step, truncate_at_step):
                    memories_grad.zero_()
                else:
                    memories_grad.copy_(segment_memories.grad.data)

            return total_loss

        # return logits if needed

        if not return_loss:
            logits = torch.cat(logits, dim = -2)
            return logits, memories

        # otherwise return losses

        return sum(losses), memories

# -----------------------------------------------------------------------------------------------

import torch
# from recurrent_memory_transformer_pytorch import RecurrentMemoryTransformer

model = RecurrentMemoryTransformer(
    num_tokens = n_dec_vocab,               # number of tokens
    num_memory_tokens = 128,          # number of memory tokens, this will determine the bottleneck for information being passed to the future
    dim = 512,                        # model dimensions
    depth = 6,                        # transformer depth
    causal = True,                    # autoregressive or not
    dim_head = 64,                    # dimension per head
    heads = 8,                        # heads
    seq_len = 1024,                   # sequence length of a segment
    use_flash_attn = True             # whether to use flash attention
)

model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# 네트워크 초기화
def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Liner층의 초기화
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# TransformerBlock모듈의 초기화 설정
model.apply(initialize_weights)

import os.path

if os.path.isfile('./checkpoints/GPT_model_Sentencepiece.pt'):
    model.load_state_dict(torch.load('./checkpoints/GPT_model_Sentencepiece.pt'))

print('네트워크 초기화 완료')

# 손실 함수의 정의
criterion = nn.CrossEntropyLoss()

# 최적화 설정
# learning_rate = 2e-4
learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

from IPython.display import clear_output
import datetime

Model_start_time = time.time()

# 학습 정의
def train(epoch, model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    accuracies = []

    with tqdm_notebook(total=len(dataloader), desc=f"Train {epoch+1}") as pbar:
        for batch_idx, samples in enumerate(dataloader):
            src_inputs, trg_outputs = samples

            # print("src_inputs  Shape :", src_inputs.shape)
            # print(src_inputs)
            mask_src = (src_inputs!=0).int()
            # print(mask_src)

            # print("trg_outputs Shape :", trg_outputs.shape)
            # print("trg_outputs :\n", trg_outputs)
            mask_trg = (trg_outputs!=0).int()
            # print(mask_trg)

            Input_concat = torch.concat((src_inputs, trg_outputs),dim=1)
            # print("Input_concat Shape :", Input_concat.shape)
            # print("Input_concat :\n", Input_concat)

            with torch.set_grad_enabled(True):

                logits1, mem1, xl_mem1   = model(Input_concat)                               # (1, 1024, 20000), (1, 128, 512), [(2, 1, 512, 512)]
                logits2, mem2, xl_mem2   = model(Input_concat, mem1, xl_memories = xl_mem1)  # (1, 1024, 20000), (1, 128, 512), [(2, 1, 512, 512)]
                logits_lm, mem3, xl_mem3 = model(Input_concat, mem2, xl_memories = xl_mem2)  # (1, 1024, 20000), (1, 128, 512), [(2, 1, 512, 512)]

                # Transformer에 입력
                # logits_lm = model(Input_concat)
                # print("logits_lm  Shape :", logits_lm.shape)
                
                pad       = torch.LongTensor(trg_outputs.size(0), 1).fill_(0).to(device)
                preds_id  = torch.transpose(logits_lm,1,2)
                labels_lm = torch.cat((trg_outputs[:, 1:], pad), -1)
                # print("labels_lm Shape: \n",labels_lm.shape)
                # print("labels_lm : \n",labels_lm)

                
                labels_concat = torch.concat((src_inputs, labels_lm),dim=1)
                # print("labels_concat Shape :", labels_concat.shape)
                # print("labels_concat :\n", labels_concat)

                
                optimizer.zero_grad()
                loss = criterion(preds_id, labels_concat)  # loss 계산


                # Accuracy
                # print("preds_id  : \n",preds_id.shape)
                mask_0 = (labels_concat!=0).int()
                arg_preds_id = torch.argmax(preds_id, axis=1)
                # print("arg_preds : \n",arg_preds_id)
                # print("arg_preds : \n",arg_preds_id.shape)
                # print("mask_0    : \n",mask_0)

                accuracy_1 = torch.eq(labels_concat, arg_preds_id).int()
                # print("accuracy_1 : \n",accuracy_1)

                accuracy_2 = torch.mul(arg_preds_id, accuracy_1).int()
                # print("accuracy_2 : \n",accuracy_2)

                accuracy = torch.count_nonzero(accuracy_2) / torch.count_nonzero(mask_0)
                # print("Accuracy : ",accuracy.clone().detach().cpu().numpy())
                accuracies.append(accuracy.clone().detach().cpu().numpy())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                epoch_loss +=loss.item()

            pbar.update(1)
            # pbar.set_postfix_str(f"Loss {epoch_loss.result():.4f} Accuracy {train_accuracy.result():.4f}")
            # pbar.set_postfix_str(f"Loss {loss.result():.4f}")
    print("accuracies :", np.mean(accuracies))
    return epoch_loss / len(dataloader)

CLIP = 0.5

epoch_ = []
epoch_train_loss = []
# 네트워크가 어느정도 고정되면 고속화
torch.backends.cudnn.benchmark = True
# epoch 루프
best_epoch_loss = float("inf")

N_EPOCHS = 100

for epoch in range(N_EPOCHS):

    train_loss = train(epoch, model, dataloader, optimizer, criterion, CLIP)

    if train_loss < best_epoch_loss:
        if not os.path.isdir("checkpoints"):
            os.makedirs("checkpoints")
        best_epoch_loss = train_loss
        torch.save(model.state_dict(), './checkpoints/GPT_model_Sentencepiece.pt')

    epoch_.append(epoch)
    epoch_train_loss.append(train_loss)
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

    # print('Epoch {0}/{1} Average Loss: {2}'.format(epoch+1, N_EPOCHS, epoch_loss))
    # clear_output(wait = True)

fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white')
ax = fig.add_subplot()
ax.plot(epoch_,epoch_train_loss, label='Average loss')

ax.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

plt.show()

# Build evaluation code.

# Predict the trained model
trained_model = RecurrentMemoryTransformer(
    num_tokens = n_dec_vocab,               # number of tokens
    num_memory_tokens = 128,          # number of memory tokens, this will determine the bottleneck for information being passed to the future
    dim = 512,                        # model dimensions
    depth = 6,                        # transformer depth
    causal = True,                    # autoregressive or not
    dim_head = 64,                    # dimension per head
    heads = 8,                        # heads
    seq_len = 1024,                   # sequence length of a segment
    use_flash_attn = True             # whether to use flash attention
).to(device)

trained_model.load_state_dict(torch.load('./checkpoints/GPT_model_Sentencepiece.pt'))


def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence

def evaluate(text):
    text = preprocess_sentence(text)
    # print(text)
    text = [vocab_src.encode_as_ids(text)]
    # print(text)
    encoder_input = pad_sequences(text, maxlen=ENCODER_LEN, padding='post', truncating='post')
    # print(encoder_input)

    decoder_input = [2]   #[BOS] token is 2
    # print(decoder_input)
    
    input  = torch.tensor(encoder_input).to(device)
    output = torch.tensor([decoder_input]).to(device)

    # print("input :", input)
    # print("output:", output)

    for i in range(DECODER_LEN):
        concate_input = torch.concat((input, output),dim=1)
        
        logits1, mem1, xl_mem1     = trained_model(concate_input)                               # (1, 1024, 20000), (1, 128, 512), [(2, 1, 512, 512)]
        logits2, mem2, xl_mem2     = trained_model(concate_input, mem1, xl_memories = xl_mem1)  # (1, 1024, 20000), (1, 128, 512), [(2, 1, 512, 512)]
        predictions, mem3, xl_mem3 = trained_model(concate_input, mem2, xl_memories = xl_mem2)  # (1, 1024, 20000), (1, 128, 512), [(2, 1, 512, 512)]
        
        # Transformer에 입력
        # logits_lm = model(Input_concat)
        # print("logits_lm  Shape :", logits_lm.shape)
                
        # print("concate_input :", concate_input)
        # predictions = trained_model(concate_input)
        # print(predictions)

        predictions = predictions[:, -1:, :]
        # print(predictions)

        # PAD, UNK, START 토큰 제외
        predicted_id = torch.argmax(predictions, axis=-1)
        # print(predicted_id)
        if predicted_id== 3:
            break

        output = torch.cat((output, predicted_id),-1)
    return output

def predict(text):
    prediction = evaluate(text)[0].detach().cpu().numpy()
    prediction = prediction[1:]
    # print("Pred IDs :", prediction)

    predicted_sentence = vocab_trg.DecodeIds(prediction.tolist())
    # print(predicted_sentence)
    return predicted_sentence

for idx in (0, 1, 2, 3):
    print("Input        :", raw_src[idx])
    print("Prediction   :", predict(raw_src[idx]))
    print("Ground Truth :", raw_trg[idx],"\n")



'''
M13. [PASS] Explore the training result with test dataset
'''
    

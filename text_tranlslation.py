#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List


# In[2]:


multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

src_language = 'de'
tgt_language = 'en'

token_transform = {}
vocab_transform = {}


# In[3]:


token_transform[src_language] = get_tokenizer(tokenizer='spacy', language='de_core_news_sm')
token_transform[tgt_language] = get_tokenizer(tokenizer='spacy', language='en_core_web_sm' )


# In[4]:


def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {
        src_language: 0,
        tgt_language: 1
    }
    for from_to_tuple in data_iter:
        yield token_transform[language](from_to_tuple[language_index[language]])

unk_index, pad_index, bos_index, eos_index = 0, 1, 2, 3
special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']

for lan in [src_language, tgt_language]:
    train_data_pipe = Multi30k(split='train', language_pair=(src_language, tgt_language))
    vocab_transform[lan] = build_vocab_from_iterator(
        iterator=yield_tokens(data_iter=train_data_pipe, language=lan),
        specials= special_tokens,
        special_first=      True
    )

for lan in [src_language, tgt_language]:
    vocab_transform[lan].set_default_index(unk_index)

print('Finished. en_vocab_len:{}. de_vocab_len:{}.'.format(len(vocab_transform[tgt_language]), len(vocab_transform[src_language])))
print(vocab_transform[tgt_language](['I', 'am', 'your', 'father']))


# In[5]:


from torch import Tensor
from torch import nn
from torch.nn import Transformer
import math

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[6]:


class TokenEmBedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmBedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        self.emd_size = emb_size

    def forward(self, token: Tensor):
        return self.embedding(token.long())*math.sqrt(self.emd_size)
    
# test
# tokens = [1, 2, 3, 4, 5]
# tokens = torch.Tensor(tokens)
# print(TokenEmBedding(10, 3)(tokens))

class PositionalEncoding(torch.nn.Module):
    def __init__(self, emd_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emd_size, 2)*math.log(10000)/emd_size)
        pos = torch.arange(0, maxlen).reshape(shape=(maxlen, 1))
        pos_embedding = torch.zeros(size=(maxlen, emd_size))
        pos_embedding[:, 0::2] = torch.sin(pos*den)
        pos_embedding[:, 1::2] = torch.cos(pos*den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
    
# test positional encoding.
# tokens = [1, 2, 3, 4, 5]
# tokens = torch.Tensor(tokens)
# token_embed = TokenEmBedding(10, 4)(tokens)
# print(token_embed.shape)
# positional_encoder = PositionalEncoding(emd_size=4, dropout=0.3)
# print(positional_encoder(token_embed))


# In[7]:


class Seq2seqTransformer(torch.nn.Module):
    def __init__(self, 
                 num_of_encoder_layer: int,
                 num_of_decoder_layer: int,
                 emd_size: int,
                 n_head: int,
                 src_embed_size: int,
                 tgt_embed_size: int,
                 dim_feed_forward: int,
                 dropout: float = 0.1):
        super(Seq2seqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emd_size,
            nhead=n_head,
            num_encoder_layers=num_of_encoder_layer,
            num_decoder_layers=num_of_decoder_layer,
            dim_feedforward=dim_feed_forward,
            dropout=dropout
        )
        self.generator = nn.Linear(emd_size, tgt_embed_size)
        self.src_token_embed = TokenEmBedding(src_embed_size, emd_size)
        self.tgt_token_embed = TokenEmBedding(tgt_embed_size, emd_size)
        self.positional_encoding = PositionalEncoding(emd_size=emd_size, dropout=dropout)
    
    def forward(self, src: Tensor, tgt: Tensor, 
                src_mask: Tensor, tgt_mask: Tensor, 
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_padding_mask: Tensor):
        src_embed = self.positional_encoding(self.src_token_embed(src))
        tgt_embed = self.positional_encoding(self.tgt_token_embed(tgt))
        out = self.transformer(src_embed, tgt_embed, src_mask, tgt_mask, None,
                               src_padding_mask, tgt_padding_mask, memory_padding_mask)
        return self.generator(out)
    
    def encode(self, src, src_mask):
        src_embed = self.positional_encoding(self.src_token_embed(src))
        return self.transformer.encoder(src_embed, src_mask)
    
    def decode(self, tgt, memory, tgt_mask):
        tgt_embed = self.positional_encoding(self.tgt_token_embed(tgt))
        return self.transformer.decoder(tgt_embed, memory, tgt_mask)
    
# test seq_seqtransformer
# model = Seq2seqTransformer(6, 6, 512, 8, 512, 512, 512)
# x = torch.ones(size=(10, 10))
# pre = model(src=x,
#             tgt=x,
#             src_mask=x,
#             tgt_mask=x,
#             src_padding_mask=x,
#             tgt_padding_mask=x,
#             memory_padding_mask=x)
# pre.shape


# In[8]:


def generate_subsequent_mask(dim: int):
    mask = torch.triu(torch.ones(size=(dim, dim), device=device) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))

    return mask

# # test
# mask = generate_subsequent_mask(dim=10)
# mask
def generate_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    src_mask = torch.zeros(size=(src_seq_len, src_seq_len), device=device).type(dtype=torch.bool)
    tgt_mask = generate_subsequent_mask(dim=tgt_seq_len)

    src_padding_mask = (src == pad_index).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_index).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# test
# src = torch.Tensor([[1, 3, 4, 5, 0]])
# tgt = src
# masks = generate_mask(src, tgt)
# masks


# In[9]:


from torch.nn.utils.rnn import pad_sequence 

def sequential_transforms(*transforms):
    def func(text_input):
        for transform in transforms:
            text_input = transform(text_input)
        return text_input
    return func

def tensor_transform(token_ids: List[int]):
    return torch.cat((
        torch.tensor([eos_index]),
        torch.tensor(token_ids  ),
        torch.tensor([eos_index])
    ))

text_tranforms = {}
for lan in [src_language, tgt_language]:
    text_tranforms[lan] = sequential_transforms(
        token_transform[lan],
        vocab_transform[lan],
        tensor_transform
    )

def collate_batch(batch):
    src_batch, tgt_batch = [], []
    for src_samples, tgt_samples in batch:
        src_batch.append(text_tranforms[src_language](src_samples.rstrip('\n')))
        tgt_batch.append(text_tranforms[tgt_language](tgt_samples.rstrip('\n')))

    src_batch = pad_sequence(sequences=src_batch, padding_value=pad_index)
    tgt_batch = pad_sequence(sequences=tgt_batch, padding_value=pad_index)

    return src_batch, tgt_batch

# test
# train_dp = Multi30k(split='train', language_pair=(src_language, tgt_language))
# from torch.utils.data import DataLoader
# train_data_loader = DataLoader(dataset=train_dp, batch_size=8, collate_fn=collate_batch)
# i = 0
# for src, tgt in train_data_loader:
#         print(src)
#         if i > 5:
#              break
#         i += 1


# In[10]:


# Prepare the train and valid dataloader
from torch.utils.data import DataLoader

BATCH_SIZE = 128
train_dp = Multi30k(split='train', language_pair=(src_language, tgt_language))
valid_dp = Multi30k(split='valid', language_pair=(src_language, tgt_language))

train_data_loader = DataLoader(dataset=train_dp, batch_size=BATCH_SIZE, collate_fn=collate_batch)
valid_data_loader = DataLoader(dataset=valid_dp, batch_size=BATCH_SIZE, collate_fn=collate_batch)


# In[11]:


torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[src_language])
TGT_VOCAB_SIZE = len(vocab_transform[tgt_language])
EMD_SIZE = 512
FFN_HIDEN_DIM = 512
N_HEAD = 8
NUM_DECODERS = 3
NUM_ENCODERS = 3

transformer = Seq2seqTransformer(
    num_of_decoder_layer=NUM_ENCODERS,
    num_of_encoder_layer=NUM_DECODERS,
    emd_size=EMD_SIZE,
    n_head=N_HEAD,
    src_embed_size=SRC_VOCAB_SIZE,
    tgt_embed_size=TGT_VOCAB_SIZE,
    dim_feed_forward=FFN_HIDEN_DIM
)

for p in transformer.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)

model = transformer.to(device=device)


# In[12]:


# train
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_index)
optimizer = torch.optim.Adam(params=model.parameters(), betas=(0.9, 0.98), lr=0.0001, eps=1e-9)

def train_epoch(model, train_data_loader, loss_fn):
    model.train()
    losses = 0
    for src, tgt in train_data_loader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = generate_mask(src=src, tgt=tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask, src_pad_mask)
        tgt_out = tgt[1:, :]
        optimizer.zero_grad()
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

        losses += loss

    return losses / len(list(train_data_loader))
    
def valid_epoch(model, valid_data_loader, loss_fn):
    model.eval()
    losses = 0
    for src, tgt in valid_data_loader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = generate_mask(src=src, tgt=tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask, src_pad_mask)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        losses += loss

    return losses / len(list(valid_data_loader))


# In[53]:


from timeit import default_timer as timer

NUM_EPOCH = 18
def train():
    print('------------Begin training-------------')
    for epoch in range(1, NUM_EPOCH+1):
        start = timer()
        train_loss = train_epoch(model=model, train_data_loader=train_data_loader, loss_fn=loss_fn)
        valid_loss = valid_epoch(model=model, valid_data_loader=valid_data_loader, loss_fn=loss_fn)
        end_time = timer()
        print(f"Epoch: {epoch}, Training loss:{train_loss:.3f}, Valid: {valid_loss:.3f}, Time = {end_time-start:.3f}s.")

train()


# In[54]:


def greedy_decode(model: torch.nn.Module, src, src_mask, max_len=100, start_symbol=bos_index):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(size=(1, 1)).fill_(start_symbol).long().to(device)
    for i in range(max_len):
        memory = memory.to(device)
        tgt_mask = generate_subsequent_mask(dim=ys.shape[0]).type(torch.bool).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)

        prob = model.generator(out[:, -1])
        _, next_token = torch.max(prob, dim=1)

        ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(src.data).fill_(next_token.squeeze(dim=0))], dim=0
        )

        if next_token == eos_index:
            break

    return ys

def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_tranforms[src_language](src_sentence.rstrip('\n')).view(-1, 1)
    mask_dim = src.shape[0]
    src_mask = (torch.zeros(size=(mask_dim, mask_dim)).type(torch.bool))
    ys = greedy_decode(model=model, src=src, src_mask=src_mask)

    sentence = []
    for token in ys:
        word = vocab_transform[tgt_language].lookup_tokens(token.cpu().numpy())
        sentence += word
        if word != '<eos>':
            sentence += ' '

    return "".join(sentence)


# In[72]:


translate(model, "Eine Gruppe von Menschen steht vor einem Iglu .")


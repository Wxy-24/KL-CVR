import math
import pickle
import re

import torch
import torch.nn as nn

from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torchvision.models import resnet18
from torchvision import transforms

from torch.nn.utils.rnn import pad_sequence


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        res = resnet18(pretrained = False)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = res.bn1
        self.relu = res.relu
        self.maxpool = res.maxpool
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.layer4 = res.layer4
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.learned_pool =  nn.Linear(960,512)
    def forward(self,image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        fv1 = self.layer1(x)
        fv2 = self.layer2(fv1)
        fv3 = self.layer3(fv2)
        fv4 = self.layer4(fv3)
        return self.learned_pool(torch.cat( [self.pool(fv).squeeze() for fv in [fv1,fv2,fv3,fv4]], dim=0))

class ContrastiveModel(nn.Module):
    def __init__(self,image_encoder,text_encoder):
        super(ContrastiveModel,self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def encode_image(self,image):
        return self.image_encoder(image)

    def encode_text(self,text):
        return self.text_encoder(text)
    
    def forward(self,image,text):
        return self.encode_image(image),self.encode_text(text)

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0),:])

class TransformerEncoderModel(nn.Module):
    def __init__(self, n_token, d_model, nhead, output_size=(128,1), dim_feedforward=2048, n_transformer_layers=2, dropout=0.5):
        """Un encoder qui a l'architecture d'un tranformer

        Parameters
        ----------
        n_token : int
            nombre de mots dans le vocabulaire
        d_model : int
            le nombre d'entrée dans le modele, la taille de l'espace d'embedding
        nhead : int 
            nombre de tete d'attention
        n_transformer_layers : int, optional
            nombre de transformer, par défaut 2
        dim_feedforward : int, optional
            la taille du réseau feedforward, par défaut 2048
        dropout : float, optional
            la valeur du dropout, par défaut 0.5
        """
        super(TransformerEncoderModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.d_model = d_model
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_transformer_layers)
        self.encoder = nn.Embedding(n_token, d_model)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src, src_mask)
        return self.avgpool(src)

class Word2Tensor():
    def __init__(self, device='cpu'):
        self.dict={}
        self.counter=1
        self.device=device

    def normalize_phrase(self, phrase):
        return re.sub('[^0-9a-zA-Z]+', ' ', phrase.lower().rstrip(' ').lstrip(' ')).split(' ')

    def update_phrase(self,phrase):
        phrase = self.normalize_phrase(phrase)
        for word in phrase:
            self.update(word)

    def update(self,word):
        if word not in self.dict.keys():
            self.dict[word]=self.counter
            self.counter+=1
    
    def translate(self, phrase):
        return torch.LongTensor([self.dict[word] for word in self.normalize_phrase(phrase)]).to(self.device)

    def translate_batch(self,batch):
        return pad_sequence([torch.LongTensor([self.dict[word] for word in self.normalize_phrase(phrase)]) for phrase in batch], batch_first=True, padding_value=0).to(self.device)
        
    def load(self,file_path):
        with open(file_path, 'rb') as f:
            self.dict = pickle.load(f)

    def save(self, file_path):
        with open(file_path,'wb') as f:
            pickle.dump(self.dict, f, pickle.HIGHEST_PROTOCOL)
    

    def translate_and_update(self, phrase):
        phrase = self.normalize_phrase(phrase)
        for word in phrase:
            self.update(word)
        return torch.LongTensor([self.dict[word] for word in phrase]).to(self.device)

    def __len__(self):
        return len(self.dict)
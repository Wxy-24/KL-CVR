import os
import re
import json
import random
import pandas as pd
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import Counter
from OpenKE.train_transe import train_transe



import spacy
import scispacy
from scispacy.linking import EntityLinker
spacy.prefer_gpu()
nlp = spacy.load("en_core_sci_scibert")
#nlp = spacy.load("/home/wxy/downloads/en_core_sci_scibert-0.5.0/en_core_sci_scibert/en_core_sci_scibert-0.5.0")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")


def parse_a_text(text):
    entities = []
    doc = nlp(text)
    for ent in doc.ents:
        # Noise Filtering
        if len(ent.text) == 1:
            continue
        # Link to UMLS
        if len(ent._.kb_ents) == 0:
            continue
        start_id = ent.start_char
        end_id = ent.end_char
        cuis = ent._.kb_ents
        cuis = [cui[0] for cui in cuis if cui[1] >= 0.95]
        if len(cuis) == 0:
            continue
        entities.append((start_id, end_id, ent.text, cuis[0]))
    return entities


def extract_umls(data):
    for split in ["train", "val", "test"]:
        split_data = data[split]
        for sample in tqdm(split_data):
            image_entites = []
            text_entities = []
            for text in sample["texts"]:
                entities = parse_a_text(text)
                text_entities.append(entities)
                image_entites.extend(entities)
            sample["image_entities"] = image_entites
            sample["text_entities"] = text_entities
    return data



def create_entity_vocab(threshold=5):
    def str2list(c):
        if '[]' in c:
            return []
        if c.count(',')==0:
            return [c[2:10]]
        else:
            cui=[]
            d=c.strip()[2:-2].split("', '")
            for i in d:
                cui.append(i)
            return cui

    df_train=pd.read_csv(f'/home/wxy/Desktop/NEW-ROCO-train.csv')
    df_valid=pd.read_csv(f'/home/wxy/Desktop/NEW-ROCO-valid.csv')
    cui_list=[]
    for i in range(len(df_train)):
        cui_list.append(df_train.iloc[i,2])
    for i in range(len(df_valid)):
        cui_list.append(df_valid.iloc[i,2])

    entity_vocab=[]
    for x in cui_list:
        cuis=str2list(x)
        if len(cuis)!=0:
            for cui in cuis:
                entity_vocab.append(cui)
    entity_vocab=list(set(entity_vocab))
    print('len(entity_vocab)',len(entity_vocab))
    

    fin = open('knowledge/train_umls_example.txt')

    entities = []
    relations = []
    triples = []

    for line_idx, line in tqdm(enumerate(fin)):
        line_splits = line.strip().split("\t")
        if line_idx==0:
            continue
        # if 'ImageCLEF' not in line_splits[0]:    
        #   if line_splits[0] not in entity_vocab and line_splits[1] not in entity_vocab:
        #       continue
        # if line_splits[0] not in entity_vocab:
        #     continue
        # if line_splits[1] not in entity_vocab:
        #     continue
        entities.append(line_splits[0])
        entities.append(line_splits[1])
        relations.append(line_splits[3])
        triples.append((line_splits[0], line_splits[1], line_splits[3]))
    print(len(relations),len(entities))

    entity_vocab = Counter(entities)
    entity_vocab = sorted(entity_vocab.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    entity2id = {k: i for i, (k, v) in enumerate(entity_vocab)}
    relation_vocab = Counter(relations)
    relation_vocab = sorted(relation_vocab.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    relation2id = {k: i for i, (k, v) in enumerate(relation_vocab)}

    os.makedirs("knowledge", exist_ok=True)
    fout = open("knowledge/train2id.txt", "wt")
    fout.write(f"{len(triples)}\n")
    for triple in triples:
        fout.write(f"{entity2id[triple[0]]}\t{entity2id[triple[1]]}\t{relation2id[triple[2]]}\n")

    fout = open("knowledge/entity2id.txt", "wt")
    fout.write(f"{len(entity2id)}\n")
    for k, v in entity2id.items():
        if k not in linker.kb.cui_to_entity.keys():
            fout.write(f"{k}\t{v}\t{k}_unknown\n")
        else:
            fout.write(f"{k}\t{v}\t{linker.kb.cui_to_entity[k].canonical_name}\n")

    fout = open("knowledge/relation2id.txt", "wt")
    fout.write(f"{len(relation2id)}\n")
    for k, v in relation2id.items():
        fout.write(f"{k}\t{v}\n")

    return entity2id, relation2id



def main_roco():
    if not (os.path.exists("knowledge/train2id.txt") and
            os.path.exists("knowledge/entity2id.txt") and
            os.path.exists("knowledge/relation2id.txt")):
        entity2id, relation2id = create_entity_vocab()
    else:
        entity2id, relation2id = {}, {}
        for line_idx, line in enumerate(open("knowledge/entity2id.txt")):
            if line_idx == 0:
                continue
            line = line.strip().split("\t")
            entity2id[line[0]] = int(line[1])
        for line_idx, line in enumerate(open("knowledge/relation2id.txt")):
            if line_idx == 0:
                continue
            line = line.strip().split("\t")
            relation2id[line[0]] = int(line[1])

    if not os.path.exists("knowledge/ent_embeddings.ckpt"):
        train_transe()

def select_image_node(ckpt_name):
    x = nn.Parameter(torch.load(f"knowledge/{ckpt_name}",map_location="cpu")["ent_embeddings.weight"], requires_grad=True)  
    x = x / x.norm(dim=1, keepdim=True)    
    emb=x.detach().numpy()
    print(emb.shape)
    f=open(f'knowledge/entity2id.txt','r')
    p=f.readlines()
    ent2idx,img_node={},[]
    for idx,i in enumerate(p):
        if idx!=0:
            ent,num=i.split('\t')[0],i.split('\t')[1]
            ent2idx[ent]=int(num)
            if len(ent)!=8 and ent[0]!='C':   # select image entity
                img_node.append(ent)
    kg_emb={}
    for img in img_node:
        idx=ent2idx[img]
        kg_emb[img]=emb[idx].tolist()

    fname=ckpt_name.replace(".ckpt",".pkl")
    with open(f'knowledge/image_node_embeddings.pkl', 'wb') as fout:
        pickle.dump(kg_emb, fout)


if __name__ == '__main__':
    main_roco()          #generate entity2id.txt/relation2id.txt/train2id.txt
    train_transe()
    select_image_node('ent_embeddings.ckpt')
    

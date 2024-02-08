import os
import re
import json
import random
import pandas as pd
import spacy
import pickle
import scispacy
from scispacy.linking import EntityLinker
from tqdm import tqdm
from collections import Counter
from OpenKE.train_transe import train_transe



spacy.prefer_gpu()
nlp = spacy.load("en_core_sci_scibert")
# nlp = spacy.load("/home/wxy/downloads/en_core_sci_scibert-0.5.0/en_core_sci_scibert/en_core_sci_scibert-0.5.0")
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
    

    fin = open('/home/wxy/work/ARL-master/data/pretrain_data/train_umls+NewRoco.txt')

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

    os.makedirs("data/knowledge/", exist_ok=True)
    fout = open("/home/wxy/work/KL-CVR/data/knowledge/train2id.txt", "wt")
    fout.write(f"{len(triples)}\n")
    for triple in triples:
        fout.write(f"{entity2id[triple[0]]}\t{entity2id[triple[1]]}\t{relation2id[triple[2]]}\n")

    fout = open("/home/wxy/work/KL-CVR/data/knowledge/entity2id.txt", "wt")
    fout.write(f"{len(entity2id)}\n")
    for k, v in entity2id.items():
        if k not in linker.kb.cui_to_entity.keys():
            fout.write(f"{k}\t{v}\t{k}_unknown\n")
        else:
            fout.write(f"{k}\t{v}\t{linker.kb.cui_to_entity[k].canonical_name}\n")

    fout = open("/home/wxy/work/KL-CVR/data/knowledge/relation2id.txt", "wt")
    fout.write(f"{len(relation2id)}\n")
    for k, v in relation2id.items():
        fout.write(f"{k}\t{v}\n")

    return entity2id, relation2id



def main_roco():
    if not (os.path.exists("data/knowledge/train2id.txt") and
            os.path.exists("data/knowledge/entity2id.txt") and
            os.path.exists("data/knowledge/relation2id.txt")):
        entity2id, relation2id = create_entity_vocab()
    else:
        entity2id, relation2id = {}, {}
        for line_idx, line in enumerate(open("data/knowledge/entity2id.txt")):
            if line_idx == 0:
                continue
            line = line.strip().split("\t")
            entity2id[line[0]] = int(line[1])
        for line_idx, line in enumerate(open("data/knowledge/relation2id.txt")):
            if line_idx == 0:
                continue
            line = line.strip().split("\t")
            relation2id[line[0]] = int(line[1])

    if not os.path.exists("data/knowledge/ent_embeddings.ckpt"):
        train_transe()




if __name__ == '__main__':
    main_roco()  

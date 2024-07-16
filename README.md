# KL-CVR
This is the implementation of [ISBI24]:[Integrating expert knowledge with vision-language model to improve medical image retrieval](https://drive.google.com/file/d/1KeQCvL60xeEMh8GhiSL_v9z7xBQxR3Bj/view?usp=drive_link)

## Requirements

Run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

### Pre-processing

Run the following command to translate knowledge graph to embeddings:

```angular2
python prepro/prepro_pretraining_data.py
```

to get the following files:

```angular2
root:
+--knowledge
| +--train2id.txt
| +--relation2id.txt
| +--entity2id.txt
| +--ent_embeddings.ckpt
| +--image_node_embeddings.pkl

```

### Fine-Tuning

Now you can start to fine-tune the model from pulicly available weights pretrained by [OpenAI](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt):

```angular2
python main.py
```

### model checkpoint

We fine-tune our framework on [ROCO](https://github.com/razorx89/roco-dataset). Our weights(ViT-Base-16) are available here [Google drive](https://drive.google.com/drive/folders/1tavJ3Xsp57ezpmzLOkfhUbTBrAt6frZv?usp=drive_link). 

You can find an example of how to use our model in this file: [jupyter notebook](https://github.com/Wxy-24/KL-CVR/blob/main/how_to_load_model.ipynb)

## Acknowledgement

The code is based on [OpenKE](https://github.com/thunlp/OpenKE), [CLIP](https://github.com/OpenAI/CLIP).
We thank the authors for their open-sourced code and encourage users to cite their works when applicable.

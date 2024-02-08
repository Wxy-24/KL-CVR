# KL-CVR
This is the implementation of [ISBI24]:Integrating expert knowledge with vision-language model to improve medical image retrieval

## Requirements

Run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

### Pre-processing

Run the following command to pre-process the data:

```angular2
python prepro/prepro_pretraining_data.py
```

to get the following files:

```angular2
root:[data]
+--knowledge
| +--train2id.txt
| +--relation2id.txt
| +--entity2id.txt
| +--ent_embeddings.ckpt

```

### Fine-Tuning

Now you can start to fine-tune the model from pulicly available weights[ViT-Base-16]([https://github.com/thunlp/OpenKE](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)):

```angular2
python main.py
```

## Acknowledgement

The code is based on [OpenKE](https://github.com/thunlp/OpenKE), [CLIP](https://github.com/OpenAI/CLIP).
We thank the authors for their open-sourced code and encourage users to cite their works when applicable.

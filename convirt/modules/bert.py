from transformers import AutoModel, AutoTokenizer,BioGptModel


def get_bert(name, freeze_layers=None):  # pretrained=True
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        if 'biogpt' in name:
            bert = BioGptModel.from_pretrained(name)
            print('biogpt.architecture:')
            print(bert.base_model.layer_norm)
        else:
            bert = AutoModel.from_pretrained(name)
            if freeze_layers:
                for idx in freeze_layers:
                    for parameter in bert.encoder.layer[idx].parameters():
                        parameter.requires_grad = False
    except OSError:
        raise KeyError(f"{name} is not a valid BERT version")
    return bert, tokenizer

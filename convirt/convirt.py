import torch
import torch.nn as nn
import numpy as np
# import clip
# import torchvision

# from simclr.modules.resnet_hacks import modify_resnet_model
# from simclr.modules.identity import Identity


class ConVIRT(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self,
                 image_encoder,
                 text_encoder,
                 # tokenizer,
                 projection_dim,
                 image_n_features,
                 text_n_features,
                 projection=False):
        super(ConVIRT, self).__init__()

        self.projection = projection

        self.image_encoder = image_encoder
        self.image_n_features = image_n_features

        self.text_encoder = text_encoder
        self.text_n_features = text_n_features

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        if self.projection:
            self.transformer_width = self.text_encoder.pooler.dense.out_features
            self.text_projection = nn.Parameter(torch.empty(self.transformer_width, projection_dim))
            nn.init.normal_(self.text_projection, std=self.transformer_width ** -0.5)

            self.image_projection = nn.Parameter(torch.empty(self.image_n_features, projection_dim))
            nn.init.normal_(self.image_projection, std=self.image_n_features ** -0.5)
        else:
            self.image_projector = nn.Sequential(
                nn.Linear(self.image_n_features, self.image_n_features, bias=False),
                nn.ReLU(),
                nn.Linear(self.image_n_features, projection_dim, bias=False),
                )

            self.text_projector = nn.Sequential(
            nn.Linear(self.text_n_features, self.text_n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.text_n_features, projection_dim, bias=False),
            )

        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        Reference: https://www.sbert.net/docs/usage/computing_sentence_embeddings.html
        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_text(self, x):
        # return self.mean_pooling(self.text_encoder(**x),
        #                      x['attention_mask']).float()
        return self.text_encoder(**x).pooler_output.cuda().float()

    def encode_image(self, x):
        return self.image_encoder(x).cuda().float()

    def forward(self, x_v, x_u):
        h_v = self.encode_image(x_v)
        h_u = self.encode_text(x_u)

        if self.projection:
            v = h_v @ self.image_projection
            u = h_u @ self.text_projection
        else:
            v = self.image_projector(h_v)
            u = self.text_projector(h_u)

        #image_features = self.image_projector(self.encode_image(x_v))
        #text_features = self.text_projector(self.encode_text(x_u))

        # normalized features
        #image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        #logit_scale = self.logit_scale.exp()
        #logits_per_image = logit_scale * image_features @ text_features.t()
        #logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        #return logits_per_image, logits_per_text #image_features, text_features,


        return v, u #h_v, h_u,

#class CLIP(nn.Module):
#    def __init__(self, model, projection_dim):
#        super(CLIP, self).__init__()
#        self.model = model

#        self.image_projector = nn.Sequential(
#            nn.Linear(1024, 1024, bias=False),
#            nn.ReLU(),
#            nn.Linear(1024, projection_dim, bias=False),
#        ).cuda()

#        self.text_projector = nn.Sequential(
#            nn.Linear(1024, 1024, bias=False),
#            nn.ReLU(),
#            nn.Linear(1024, projection_dim, bias=False),
#        ).cuda()

#    def forward(self, x_v, x_u):
#        h_v = self.model.encode_image(x_v).cuda()
#        h_u = self.model.encode_text(x_u).cuda()

#        v = self.image_projector(h_v.float())
#        u = self.text_projector(h_u.float())
#        return h_v, h_u, v, u

class ConVIRT_CLIP(nn.Module):
    def __init__(self,
                 image_encoder,
                 text_encoder):
        super(ConVIRT_CLIP, self).__init__()

        self.image_encoder = image_encoder

        self.dtype = self.image_encoder.conv1.weight.dtype

        self.embed_dim = self.image_encoder.output_dim

        self.text_encoder = text_encoder

        # self.text_projector = nn.Sequential(
        #     nn.Linear(768, 768, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(768, 640, bias=False),
        # )
        self.transformer_width = self.text_encoder.pooler.dense.out_features

        self.text_projection = nn.Parameter(torch.empty(self.transformer_width, self.embed_dim))
        nn.init.normal_(self.text_projection, std=self.transformer_width ** -0.5)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_text(self, x):
        return self.text_encoder(**x).pooler_output @ self.text_projection

    def encode_image(self, x):
        return self.image_encoder(x.type(self.dtype)) # .float()

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text #image_features, text_features,

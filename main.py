import os
import numpy as np
import torch
import shutil
import torchvision
import argparse
import datetime
import time
import pickle
import torch.nn.functional as F
from torch import nn
torch.autograd.set_detect_anomaly(True)
from transformers import AutoTokenizer

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# ConVIRT
import clip
from modules import NT_Xent, get_bert, ContrastiveLoss
from modules.transformations import TransformsConVIRT
from modules.sync_batchnorm import convert_model
from modules.dataloader import CLRDataset
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from model import load_optimizer, save_model
from utils import yaml_config_hook

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad != None:
            p.grad.data = p.grad.data.float()
def convert_models_to_mix(model):
    clip.model.convert_weights(model)


class KL_CVR(nn.Module):
    def __init__(self, model):
        super(KL_CVR, self).__init__()
        self.model = model
        self.temp_itc=model.logit_scale
        self.temp_ikc=nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.img_align = nn.Parameter(torch.empty(768, 512))
        nn.init.normal_(self.img_align, std=512 ** -0.5)


    def forward(self, x_v, x_u, x_k):
        image_emb = self.model.encode_image(x_v)
        image_features = image_emb @ self.model.visual.proj
        text_features = self.model.encode_text(x_u)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)


        # cosine similarity as logits
        logits_per_image = self.temp_itc.exp()* image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        #alignment
        aligned_image_features = image_emb @ self.img_align
        aligned_image_features = aligned_image_features / aligned_image_features.norm(dim=1, keepdim=True)
        align_matrix_img=self.temp_ikc.exp() * aligned_image_features @ x_k.t()


        return logits_per_image, logits_per_text,align_matrix_img

def train(args, train_loader, model, tokenizer, criterion, optimizer,writer):
    loss_epoch = 0
    for step, (images,texts,x_n) in enumerate(train_loader):
        optimizer.zero_grad()
        x_v = images.to(args.device)
        if args.emb!=0:
            img_emb=[]
            for img_name in x_n:
                img_key=img_name
                if img_key in emb.keys():
                    img_emb.append(emb[img_key])
                else:
                    img_emb.append(np.array([0.01]*512))
        align_emb=torch.tensor(img_emb).float().cuda()
        align_emb = align_emb / align_emb.norm(dim=1, keepdim=True)



        x_u = tokenizer(texts, truncate=True).to(args.device)
        v, u,align_matrix_img = model(x_v, x_u,align_emb)


        labels = torch.arange(args.batch_size, dtype=torch.long, device=args.device)
        loss_img = torch.nn.CrossEntropyLoss()
        loss_txt = torch.nn.CrossEntropyLoss()
        loss_itc= loss_img(v,labels)+loss_txt(u,labels)
        loss_ikc= loss_img(align_matrix_img,labels)+loss_txt(align_matrix_img.t(),labels)
        loss = loss_itc+2*loss_ikc


        loss.backward()
        # convert_models_to_fp32(model)
        optimizer.step()
        # convert_models_to_mix(model)

        if args.nr == 0 and step % 1000 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


def validate(args, val_loader, model, tokenizer, criterion, optimizer,writer):
    with torch.no_grad():
        model.eval()
        loss_epoch = 0
        for step, (x_v, x_u,x_n) in enumerate(val_loader):
            x_v = x_v.to(args.device)
            if args.emb!=0:
                img_emb=[]
                for img_name in x_n:
                    img_key=img_name
                    if img_key in emb.keys():
                        img_emb.append(emb[img_key])
                    else:
                        img_emb.append(np.array([0.01]*512))
            align_emb=torch.tensor(img_emb).float().cuda()
            align_emb = align_emb / align_emb.norm(dim=1, keepdim=True)



            x_u = tokenizer(x_u, truncate=True).to(args.device)
            v, u,align_matrix_img = model(x_v, x_u,align_emb)

            labels = torch.arange(args.batch_size, dtype=torch.long, device=args.device)
            loss_img = torch.nn.CrossEntropyLoss()
            loss_txt = torch.nn.CrossEntropyLoss()
            loss_itc= loss_img(v,labels)+loss_txt(u,labels)
            loss_ikc= loss_img(align_matrix_img,labels)+loss_txt(align_matrix_img.t(),labels)
            loss = loss_itc+2*loss_ikc

            loss_epoch += loss.item()

    model.train()
    return loss_epoch



def main(gpu, args):

    shutil.copy('./config/config.yaml',
                os.path.join(args.model_path, 'config.yaml'))

    is_master=1
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ### MOVED ###
    if "clip" in args.vit:
        # initialize model
        print("Initializing model... ", end="", flush=True)
        model, preprocess = clip.load(args.vit.split("@")[-1], device=args.device, jit=False)
        model = model.float()
        tokenizer = clip.tokenize

        if "align" in args.vit:
            print('image knowledge projection!')
            model = KL_CVR(model)
        else:
            print('default CLIP')
            args.emb=0

        print("Image encoder:", args.vit,
            "\tPretrained:", args.pretrain)
        print("Tokenizer:",tokenizer)


    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
        )
        model.load_state_dict(torch.load(model_fp,
                                         map_location=args.device.type))

    model = model.to(args.device)

    # optimizer / loss
    print("Loss:", args.criterion, "\t")
    optimizer, scheduler = load_optimizer(args, model, lr=float(args.lr))
    if args.criterion == "contrastive":
        criterion = ContrastiveLoss(args.batch_size, args.device)
    else:
        raise NotImplementedError


   ### MOVED ###

    if "clip" in args.vit or "RN50" in args.vit:
        transform = preprocess
    else:
        transform = TransformsConVIRT(size=args.image_size,
                                      sampling=args.sampling)

    train_dataset = CLRDataset(csv_file=args.csv_file,
                               root_dir=args.root_dir,
                               transform=transform,
                               clip = ("clip" in args.vit or "RN50" in args.vit)
                               )
    print('training set:',args.csv_file,'len(dataset):',train_dataset.__len__())

    val_dataset = CLRDataset(csv_file=args.val_csv_file,
                             root_dir=args.val_root_dir,
                             transform=transform,
                             clip = ("clip" in args.vit or "RN50" in args.vit)
                             )
    print('validation set:',args.val_csv_file,'len(csv):',val_dataset.__len__())
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers
    )

    if is_master:
        print("[DONE]\n")

    # print(list(train_loader)[0])

    writer = None
    if args.nr == 0:
        writer = SummaryWriter()

    if is_master:
        print("STARTING TRAINING")
        print('Start Time =', datetime.datetime.now().strftime("%H:%M:%S"), '\n')

    t0 = time.time()
    args.global_step = 0
    args.current_epoch = 0
    best_val_loss = np.inf


    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, tokenizer, criterion, optimizer,writer)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 10 == 0 and is_master:
            save_model(args, model, optimizer)

        if args.nr == 0 and is_master:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            val_loss = validate(args, val_loader, model, tokenizer, criterion, optimizer,writer)
            if val_loss < best_val_loss:
                save_model(args, model, optimizer, best=True)
                best_val_loss = val_loss
            else:
                save_model(args, model, optimizer, best=False)

            epoch_counter = epoch - args.start_epoch
            elapsed = time.time() - t0
            epoch_time = elapsed/(epoch_counter+1)
            remaining = (args.epochs - (epoch_counter+1))*epoch_time
            remaining = str(datetime.timedelta(seconds=round(remaining)))
            elapsed = str(datetime.timedelta(seconds=round(elapsed)))
            print(f'Epoch {epoch_counter+1}/{args.epochs} [{elapsed}<{remaining}, {round(epoch_time, 2)}s/epoch] {round((epoch_counter+1)/args.epochs*100, 1)}% loss: {loss_epoch / len(train_loader)}\t val_loss: {val_loss / len(val_loader)} lr: {lr}')

            log_filename='training_logfile'
            with open(os.path.join(args.log_loss_dir,log_filename), 'a') as f:
                f.write(str(loss_epoch / len(train_loader)) + ',' + str(val_loss / len(val_loader)) + '\n')

            args.current_epoch += 1

    # end training
    if is_master:
        save_model(args, model, optimizer)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConVIRT")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    print("args.model_path",args.model_path)
    with open(args.pkl_file, 'rb') as f:
        emb = pickle.load(f)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", args.device)
    main(0, args)


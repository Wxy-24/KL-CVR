import torchvision
import random
from PIL import Image


class TransformsConVIRT:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size, sampling=False, test=False):

        self.sampling = sampling
        self.test = test

        color_jitter = torchvision.transforms.ColorJitter(
            brightness=(0.6, 1.4), contrast=(0.6, 1.4)
        )
        #self.train_transform = torchvision.transforms.Compose(
        #    [
        #        torchvision.transforms.RandomResizedCrop(size=size,
        #                                                 ratio=(0.6, 1.0)),
        #        torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
        #        torchvision.transforms.RandomAffine(degrees=(-20, 20),
        #                                            translate=(0.1, 0.1),
        #                                            scale=(0.95, 1.05)),
        #        torchvision.transforms.RandomApply([color_jitter], p=0.8),
        #        # torchvision.transforms.RandomGrayscale(p=0.2),
        #        torchvision.transforms.GaussianBlur(kernel_size=3,
        #                                            sigma=(0.1, 3.0)),
        #        torchvision.transforms.ToTensor(),
        #    ]
        #)
        self.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size, interpolation=Image.BICUBIC),
            torchvision.transforms.CenterCrop(size),
            lambda image: image.convert("RGB"),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=[size,size]),
                torchvision.transforms.ToTensor(),
            ]
        )

    def text_sampling(self, text):
        text = text.replace("\n", " ")
        if self.sampling:
            text = text.split(".")
            if '' in text:
                text.remove('')
            text = random.choice(text)
        return text

    def __call__(self, x):
        if self.test:
            return self.test_transform(x['image']), self.text_sampling(x['text'])
        else:
            return self.train_transform(x['image']), self.text_sampling(x['text'])

class TransformsIRMA:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size, sampling=False, test=False):

        self.sampling = sampling
        self.test = test

        color_jitter = torchvision.transforms.ColorJitter(
            brightness=(0.6, 1.4), contrast=(0.6, 1.4)
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size,
                                                         ratio=(0.6, 1.0)),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomAffine(degrees=(-20, 20),
                                                    translate=(0.1, 0.1),
                                                    scale=(0.95, 1.05)),
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                # torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.GaussianBlur(kernel_size=3,
                                                    sigma=(0.1, 3.0)),
                torchvision.transforms.ToTensor(),
            ]
        )

        #self.test_transform = torchvision.transforms.Compose(
        #    [
        #        torchvision.transforms.Resize(size=[size,size]),
        #        torchvision.transforms.ToTensor(),
        #    ]
        #)
        self.test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size, interpolation=Image.BICUBIC),
        torchvision.transforms.CenterCrop(size),
        lambda image: image.convert("RGB"),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    def __call__(self, x):
        return self.test_transform(x)

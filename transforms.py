




import torchvision.transforms as transforms
from PIL import ImageOps

def get_transform(opt, phase):

    t_list = [PadSquare(), transforms.Resize(256)]

    if phase == 'train':

        t_list += [transforms.RandomHorizontalFlip(), transforms.RandomCrop(opt.crop_size)]
    else:
        t_list += [transforms.CenterCrop(opt.crop_size)]

    t_list += [transforms.ToTensor(), Normalizer()]
    return MyTransforms(t_list)


class PadSquare(object):

    def __call__(self, img):

        w, h = img.size
        if w > h:
            delta = w - h
            padding = (0, delta//2, 0, delta - delta//2)
            img = ImageOps.expand(img, padding, (255, 255, 255))
        elif w < h:
            delta = h - w
            padding = (delta//2, 0, delta - delta//2, 0)
            img = ImageOps.expand(img, padding, (255, 255, 255))
        return img


def Normalizer():

    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

class MyTransforms(object):

    def __init__(self, trfs_list):
        self.transform = transforms.Compose(trfs_list)

    def __call__(self, x):
        y = self.transform(x)
        return y
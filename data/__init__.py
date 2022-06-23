from .dataset import Dataset
import torchvision
import torch
import numpy as np

def to_tensor():
    def _to_tensor(image):
        if len(image.shape) == 3:
            return torch.from_numpy(
                image.transpose(2, 0, 1).astype(np.float32))
        else:
            return torch.from_numpy(image[None, :, :].astype(np.float32))

    return _to_tensor


def normalize(mean, std):
    mean = np.array(mean)
    std = np.array(std)

    def _normalize(image):
        image = np.asarray(image).astype(np.float32) / 255.
        image = (image - mean) / std
        return image

    return _normalize


def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax, :] = torch.rand(mask_size, mask_size, 3)
        return image

    return _cutout


def generate_loader(phase, opt):
    dataset = Dataset
    mean, std = 0.5, 0.5
    if phase == 'train':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((opt.img_size, opt.img_size)),
            normalize(mean, std),
            cutout(opt.img_size//4, 1, True),
            to_tensor()
        ])
        dataset = dataset(opt, phase, transform=transforms)

    else :
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((opt.img_size, opt.img_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
        dataset = dataset(opt, phase, transform=transforms)
    kwargs = {
        "batch_size": opt.batch_size if phase == 'train' else opt.eval_batch_size,
        "num_workers": opt.num_workers if phase == 'train' else 0,
        "shuffle": phase == 'train',
        "drop_last": phase == 'train',
    }
    return torch.utils.data.DataLoader(dataset, **kwargs)






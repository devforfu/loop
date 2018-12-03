from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader, is_image_file

from image_renderer import RAW_SIZE, default_renderer


class Doodles(Dataset):

    def __init__(self, root: Path, train: bool = True,
                 subset_size: int = None, image_size: int = RAW_SIZE,
                 renderer=default_renderer, transforms=None):

        subfolder = root / ('train' if train else 'valid')
        if isinstance(image_size, int):
            image_size = image_size, image_size

        worker = partial(read_category, subset_size)
        with Pool(cpu_count()) as pool:
            data = pool.map(worker, subfolder.glob('*.csv'))

        merged = pd.concat(data)
        targets = merged.word.values
        classes = np.unique(targets)
        class2idx = {v: k for k, v in enumerate(classes)}
        labels = np.array([class2idx[c] for c in targets])

        self.root = root
        self.train = train
        self.subset_size = subset_size
        self.image_size = image_size
        self.renderer = renderer
        self.data = merged.drawing.values
        self.classes = classes
        self.class2idx = class2idx
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        strokes, target = self.data[item], self.labels[item]
        img = self.renderer.render(strokes, self.image_size)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target


class TestImagesFolder(Dataset):

    def __init__(self, path, image_size=RAW_SIZE,
                 loader=pil_loader, pseudolabel=0):
        path = Path(path)

        if isinstance(image_size, int):
            image_size = image_size, image_size

        assert path.is_dir() and path.exists(), 'Not a directory!'
        assert path.stat().st_size > 0, 'Directory is empty'

        images = [file for file in path.iterdir() if is_image_file(str(file))]

        self.path = path
        self.image_size = image_size
        self.loader = loader
        self.images = images
        self.pseudolabel = pseudolabel

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = self.loader(self.images[item])
        img.thumbnail(self.image_size, PIL.Image.ANTIALIAS)
        return img, self.pseudolabel


def read_category(subset_size, path):
    if subset_size is None:
        return pd.read_csv(path)

    data = pd.DataFrame()
    for chunk in pd.read_csv(path, chunksize=min(10000, subset_size)):
        data = data.append(chunk)
        if len(data) >= subset_size:
            break

    return data[:subset_size]
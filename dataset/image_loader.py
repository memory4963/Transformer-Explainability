# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from skimage.filters.rank import entropy
import os


class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - img000.png
            - img001.png
            - folder1/
                - img002.png
                - img003.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        patch_size (int): patch size of the transformer network
    """

    def __init__(self, root, transform=None, normalize=None, patch_size=7):
        root = Path(root)

        if not root.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = []
        for root, dirs, files in os.walk(root):
            for file in files:
                if file.endswith('.JPEG'):
                    self.samples.append(os.path.join(root, file))

        self.transform = transform
        self.normalize = normalize
        self.patch_size = patch_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        entr_img = entropy(img[0], np.ones([self.patch_size,self.patch_size]))
        entr_img += entropy(img[1], np.ones([self.patch_size,self.patch_size]))
        entr_img += entropy(img[2], np.ones([self.patch_size,self.patch_size]))
        entr_img = entr_img[self.patch_size//2::self.patch_size,self.patch_size//2::self.patch_size].reshape(-1)
        if self.normalize:
            img = self.normalize(img)
        return img, entr_img

    def __len__(self):
        return len(self.samples)

#%%
import numpy as np
from PIL import Image
from os.path import join, exists
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T


def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

    Args:
        data: list of tuple (query, positive, negatives). 
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)                                                 # ([B, C, H, W]) = ([C, H, W]) + ...
    positive = data.dataloader.default_collate(positive)                                           # ([B, C, H, W]) = ([C, H, W]) + ...
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])                   # ([B, C, H, W]) = ([C, H, W])
    negatives = torch.cat(negatives, 0)                                                            # ([B*n_neg, C, H, W])
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices


class Base(data.Dataset):
    """
    SCFace Dataset Base Class
    
    Expects preprocessed SCFace data structure:
    dbs/SCFace/
        ├── images/
        │   ├── PersonID_1/
        │   │   ├── cam1_img1.jpg
        │   │   └── cam2_img1.jpg
        │   └── PersonID_2/
        │       └── cam1_img1.jpg
        ├── image_class_labels.txt (format: <image_id> <person_id>)
        └── images.txt (format: <image_id> <relative_path>)
    """
    def __init__(self, split='test', data_path='', aug=False) -> None:
        # SCFace is primarily used for testing/evaluation (no training split)
        # Default aug=False since SCFace is for evaluation
        if aug:
            self.input_transform = T.Compose([
                T.ToTensor(),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.NEAREST, expand=False, center=None, fill=0),
                T.RandomResizedCrop((224, 224), scale=(0.4, 1.0), ratio=(0.75, 1.33), interpolation=T.InterpolationMode.NEAREST),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        else:
            self.input_transform = T.Compose([
                T.ToTensor(),
                T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST),
            ])

        self.dataset_dir = join(data_path, 'images')

        # get label
        image_label = []
        with open(join(data_path, "image_class_labels.txt"), 'r') as f:
            lines = f.readlines()
            for line in lines:
                label = int(line.split(' ')[1].strip())
                image_label.append(label)
        image_label = np.array(image_label)

        # SCFace split logic: typically all data is used for testing
        # Can be modified based on specific experimental setup
        if split == 'train':
            # If training split needed, use a subset
            # Example: first 80% of persons for training
            unique_labels = np.unique(image_label)
            train_labels = unique_labels[:int(0.8 * len(unique_labels))]
            selected_indices = np.where(np.isin(image_label, train_labels))[0]
        elif split == 'val':
            # Use remaining 20% for validation
            unique_labels = np.unique(image_label)
            val_labels = unique_labels[int(0.8 * len(unique_labels)):]
            selected_indices = np.where(np.isin(image_label, val_labels))[0]
        elif split == 'test':
            # Use all data for testing (typical SCFace usage)
            selected_indices = np.arange(len(image_label))
        else:
            raise NameError('undefined split')

        self.image_label = image_label[selected_indices]

        # get image list
        image_list = []
        with open(join(data_path, "images.txt"), 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_path = line.split(' ')[1].strip()
                image_list.append(img_path)
        image_list = np.array(image_list)
        self.image_list = image_list[selected_indices]

    def load_image(self, index):
        filepath = join(self.dataset_dir, self.image_list[index])
        img = Image.open(filepath)
        if img.mode != 'RGB':                      # some samples are greyscale images, convert to RGB
            img = img.convert("RGB")
        if self.input_transform:
            img = self.input_transform(img)
        return img


class Whole(Base):
    def __init__(self, split='test', data_path='', aug=False, return_label=False) -> None:
        super().__init__(split=split, data_path=data_path, aug=aug)
        self.return_label = return_label
        # get positives
        self.positives = []
        for i, label in enumerate(self.image_label):
            positive = np.where(self.image_label == label)[0]                                      # find same-label samples
            positive = np.delete(positive, np.where(positive == i)[0])                             # delete self
            self.positives.append(positive)

    def __len__(self):
        return len(self.image_list)

    def load_image(self, index):
        filepath = join(self.dataset_dir, self.image_list[index])
        img = Image.open(filepath)
        if img.mode != 'RGB':                      # some samples are greyscale images, convert to RGB
            img = img.convert("RGB")
        if self.input_transform:
            img = self.input_transform(img)
        return img

    def __getitem__(self, index):
        img = self.load_image(index)
        if self.return_label:
            label = self.image_label[index]
            return img, index, label
        else:
            return img, index

    def get_positives(self):
        return self.positives


class Tuple(Base):
    """
    Tuple dataset for SCFace - primarily for evaluation with triplet-based methods
    Note: SCFace is typically used for evaluation only, not training
    """
    def __init__(self, split='test', data_path='', margin=0.5, aug=False) -> None:
        super().__init__(split=split, data_path=data_path, aug=aug)

        self.margin = margin
        self.n_negative_subset = 1000

        # get positives
        self.positives = []
        for i, label in enumerate(self.image_label):
            positive = np.where(self.image_label == label)[0]                                      # find same-label samples
            positive = np.delete(positive, np.where(positive == i)[0])                             # delete self
            self.positives.append(positive)

        # get negatives
        self.negatives = []
        for i, positive in enumerate(self.positives):
            negative = np.setdiff1d(np.arange(len(self.image_label)), positive, assume_unique=True)
            negative = np.delete(negative, np.where(negative == i)[0])                             # delete self
            self.negatives.append(negative)

        self.n_neg = 5
        self.cache = None                        # NOTE: assign a CPU tensor instead of a CUDA tensor to self.cache

    def __len__(self):
        return len(self.image_list)

    def load_image(self, index):
        filepath = join(self.dataset_dir, self.image_list[index])
        img = Image.open(filepath)
        if img.mode != 'RGB':                  # some samples are greyscale images, convert to RGB
            img = img.convert("RGB")
        if self.input_transform:
            img = self.input_transform(img)
        return img

    def __getitem__(self, index):

        # mining the closest positive
        p_indices = self.positives[index]
        a_emd = self.cache[index]                                                                  # (1,D)
        p_emd = self.cache[p_indices]                                                              # (Np, D)
        dist = torch.norm(a_emd - p_emd, dim=1, p=None)                                            # Np
        d_p, inds_p = dist.topk(1, largest=False)                                                  # choose the closest positive
        index_p = self.positives[index][inds_p].item()

        # mining the closest negative
        n_indices = self.negatives[index]
        n_emd = self.cache[n_indices]                                                              # (Np, D)
        dist = torch.norm(a_emd - n_emd, dim=1, p=None)                                            # Np
        d_n, inds_n = dist.topk(self.n_neg * 100, largest=False)                                   # choose the closest negative
        violating_indices = d_n < d_p + self.margin                                                # [True, True, ...] tensor
        if torch.sum(violating_indices) < 1:
            return None
        inds_n_vio = inds_n[violating_indices][:self.n_neg].numpy()                                # tensor -> numpy
        index_n = n_indices[inds_n_vio]

        # load images
        a_img = self.load_image(index)
        p_img = self.load_image(index_p)
        n_img = [self.load_image(ind) for ind in index_n]
        n_img = torch.stack(n_img, 0)            # (n_neg, C, H, W])

        return a_img, p_img, n_img, [index, index_p] + index_n.tolist()

    def get_positives(self):
        return self.positives


if __name__ == '__main__':
    # Test SCFace dataset loading
    whole_test_set = Whole('test', data_path='dbs/SCFace', aug=False)
    print(f"SCFace test set size: {len(whole_test_set)}")
    
    # Example: test with tuple dataset if needed
    # test_set = Tuple('test', data_path='dbs/SCFace', aug=False)
    # print(f"SCFace tuple test set size: {len(test_set)}")

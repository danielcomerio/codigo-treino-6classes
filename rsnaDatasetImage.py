from xmlrpc.client import Boolean
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pandas import DataFrame
from torch import Tensor
from os.path import join
from cv2 import imread


def generate_transforms(input_size: int, is_train: bool=True) -> transforms: # tuple[transforms] == typerror
    if not is_train:
        return (transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([input_size, input_size]),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))


    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([input_size, input_size]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=(int(input_size * 0.9), int(input_size * 0.9))),
        transforms.RandomRotation(degrees=10),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing()
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([input_size, input_size]),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms


class RsnaDatasetImage(Dataset):
    """Brain CT Dataset for Images."""

    def __init__(self,
                 dataset_path: str,
                 dataset_lines: DataFrame,
                 file_names: None, # : list[str]
                 transforms: transforms=None,
                 return_all_labels: bool=False,
                 is_train: bool=True,) -> None:
        
        self.data = []
        for row_id, val in dataset_lines.iterrows():
            if val.ID in file_names:
                
                #there_is_hemorrage = 1 if val.epidural == 1 or val.intraparenchymal == 1 or val.intraventricular == 1 or val.subarachnoid == 1 or val.subdural == 1 else 0
                #if return_all_labels:
                there_is_hemorrage = [val.epidural, val.intraparenchymal, val.intraventricular, val.subarachnoid, val.subdural, val["any"]]
                    
                self.data.append([
                    val.ID,
                    there_is_hemorrage
                ])

        if is_train:
            healthy = [d for d in self.data if d[1][5] == 0]
            hemorrhage = [d for d in self.data if d[1][5] == 1]

            balanced_data = []
            for i in range(len(healthy)):
                balanced_data.append(healthy[i])
                balanced_data.append(hemorrhage[i % len(hemorrhage)])

            self.data = balanced_data
        self.dataset_path = dataset_path
        self.transforms = transforms


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, idx: int) -> None: # tuple[Tensor, int]
        image_name, hemo_label = self.data[idx]

        image_path = join(self.dataset_path, image_name + ".png") 
        image = imread(image_path)

        if self.transforms:
            image = self.transforms(image)

        return image_name, image, hemo_label


def generate_dataset(dataset_path: str,
                     dataset_lines: DataFrame,
                     file_names: None, # list[str]
                     transforms: transforms,
                     val_file_names: None=None, # list[str]=None
                     val_transforms: transforms=None,
                     return_all_labels: bool=False,
                     is_train: bool=True) -> None: # tuple[Dataset]
    
    if not is_train:
        return (RsnaDatasetImage(dataset_path,
                                 dataset_lines,
                                 file_names,
                                 transforms=transforms,
                                 return_all_labels=return_all_labels,
                                 is_train=False))

    train_dataset = RsnaDatasetImage(dataset_path,
                                     dataset_lines,
                                     file_names,
                                     transforms=transforms,
                                     return_all_labels=return_all_labels,
                                     is_train=True)

    val_dataset = RsnaDatasetImage(dataset_path,
                                   dataset_lines,
                                   val_file_names,
                                   transforms=val_transforms,
                                   return_all_labels=return_all_labels,
                                   is_train=False)

    return train_dataset, val_dataset


def generate_loader(dataset: Dataset,
                    batch_size: int=64,
                    val_dataset: Dataset=None,
                    val_batch_size: int=64,
                    workers: int=2,
                    is_train=True) -> None: # tuple[DataLoader]
    
    if not is_train:
        return (DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=workers))

    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=workers)

    return train_loader, val_loader
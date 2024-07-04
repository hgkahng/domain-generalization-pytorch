
import os
import typing

import numpy as np
import pandas as pd

import torch
import lightning as L

from torch.utils.data import ConcatDataset, DataLoader
from torchvision.io import read_image, ImageReadMode

from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from rich.progress import track

from dg_lightning.datasets.base import SupervisedDataModule
from dg_lightning.datasets.base import SemiSupervisedDataModule
from dg_lightning.datasets.loaders import InfiniteDataLoader


locations = {
    'labeled': list(range(323)),             #     0 ~   323
    'unlabeled': list(range(10000, 13215)),  # 10000 ~ 13214
}


class IWildCam(torch.utils.data.Dataset):
    """
        Input (x): RGB images from camera traps
        Label (y): one of 182 classes corresponding to animal species
            In the metadata, each instance is annotated with the ID of the location
                (camera trap) it came from.
    """
    size: tuple = (448, 448)
    _domain_col: str = 'location_remapped'
    
    def __init__(self,
                 root: str = './data/wilds/iwildcam_v2.0',
                 locations: typing.Iterable[int] = locations['labeled'],
                 split: str = None,
                 metadata: typing.Optional[pd.DataFrame] = None,
                 ):
        super().__init__()

        self.root = root
        self.locations = locations
        self.split = split

        # read metadata if not provided
        if not isinstance(metadata, pd.DataFrame):
            metadata = pd.read_csv(os.path.join(self.root, 'metadata.csv'), index_col=0)

        if self.split is not None:
            if self.split not in ('train', 'id_val', 'val', 'id_test', 'test'):
                raise ValueError
            
        # filter metadata 
        rows_to_keep = metadata[self._domain_col].isin(self.locations)
        if self.split is not None:
            rows_to_keep = rows_to_keep & (metadata['split'] == self.split)
        metadata = metadata.loc[rows_to_keep].copy()
        metadata = metadata.reset_index(drop=True, inplace=False)
        self.metadata = metadata

        # main attributes
        self.input_files = [
            os.path.join(self.root, 'train', filename) for filename in metadata['filename'].values
        ]  # Note: all data instances are stored in the `train` folder
        self.targets = torch.from_numpy(metadata['y'].values).long()
        self.domains = torch.from_numpy(metadata[self._domain_col].values).long()
        self.eval_groups = self.domains.clone()

        # for resizing the RGB images on the CPU
        from torchvision.transforms.v2 import Resize
        self.resize_fn = Resize(size=self.size)

    def get_input(self, index: int) -> torch.ByteTensor:
        img = read_image(self.input_files[index], mode=ImageReadMode.RGB)
        return self.resize_fn(img)
    
    def get_target(self, index: int) -> torch.LongTensor:
        return self.targets[index]

    def get_domain(self, index: int) -> torch.LongTensor:
        return self.domains[index]
    
    def get_eval_group(self, index: int) -> torch.LongTensor:
        return self.eval_groups[index]
    
    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return dict(
            x=self.get_input(index),
            y=self.get_target(index),
            domain=self.get_domain(index),
            eval_group=self.get_eval_group(index),
        )
    
    def __len__(self) -> int:
        return len(self.metadata)

    @staticmethod
    def download(root: str) -> None:
        raise NotImplementedError

    @property
    def input_shape(self) -> typing.Tuple[int]:
        return (3, ) + self.size
    
    @property
    def num_classes(self) -> int:
        return 182


class UnlabeledIWildCam(torch.utils.data.Dataset):
    size: tuple = IWildCam.size
    _domain_col: str = IWildCam._domain_col
    def __init__(self, 
                 root: str = './data/wilds/iwildcam_unlabeled_v1.0',
                 locations: typing.Iterable[int] = locations['unlabeled'],
                 metadata: typing.Optional[pd.DataFrame] = None,
                 ) -> None:
        super().__init__()

        self.root = root
        self.locations = locations

        # Read metadata
        if not isinstance(metadata, pd.DataFrame):
            metadata = pd.read_csv(os.path.join(self.root, 'metadata.csv'), index_col=0)

        # Keep rows of metadata specific to those in `locations`
        rows_to_keep = metadata[self._domain_col].isin(self.locations)
        metadata = metadata.loc[rows_to_keep].copy()
        metadata = metadata.reset_index(drop=True, inplace=False)

        metadata['filename'] = metadata['uid'].apply(lambda x: x + '.jpg')
        self.metadata = metadata

        # Main attributes
        self.input_files = metadata['filename'].tolist()
        self.domains = torch.from_numpy(metadata[self._domain_col].values).long()
        self.eval_groups = self.domains.clone()
        
        # For resizing the RGB images on the CPU
        from torchvision.transforms.v2 import Resize
        self.resize_fn = Resize(size=self.size)

    def get_input(self, index: int) -> torch.ByteTensor:
        img_path = os.path.join(self.root, 'images', self.input_files[index])
        try:
            img = read_image(img_path, mode=ImageReadMode.RGB)
        except RuntimeError:
            img = pil_to_tensor(Image.open(img_path))
        return self.resize_fn(img)
        
    def get_domain(self, index: int) -> torch.LongTensor:
        return self.domains[index]
    
    def get_eval_group(self, index: int) -> torch.LongTensor:
        return self.eval_groups[index]

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return {
            'x': self.get_input(index),
            'domain': self.get_domain(index),
            'eval_group': self.get_eval_group(index),
        }
    
    def __len__(self) -> int:
        return len(self.input_files)
    
    @property
    def input_shape(self) -> typing.Tuple[int]:
        return (3, ) + self.size
    

class IWildCamDataModule(SupervisedDataModule):
    _domain_col = IWildCam._domain_col
    def __init__(self, 
                 root: str = './data/wilds/iwildcam_v2.0',
                 batch_size: typing.Optional[int] = 16,
                 num_workers: typing.Optional[int] = 8,
                 prefetch_factor: typing.Optional[int] = 4,
                 ) -> None:

        super().__init__()

        # dataset arguments
        self.root = root

        # dataloader arguments
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        # read metadata
        self.metadata = pd.read_csv(os.path.join(self.root, 'metadata.csv'), index_col=0)

        # dictionary of list of domains
        domain_splits = self.get_domain_splits(self.metadata)

        # lists of domains
        self.train_domains = domain_splits['train']
        self.id_validation_domains = domain_splits['id_val']
        self.ood_validation_domains = domain_splits['ood_val']
        self.id_test_domains = domain_splits['id_test']
        self.ood_test_domains = domain_splits['ood_test']

        # alias for consistency
        self.validation_domains = self.ood_validation_domains
        self.test_domains = self.ood_test_domains

        # check
        assert all([d in self.train_domains for d in self.id_validation_domains])
        assert all([d in self.train_domains for d in self.id_test_domains])
        assert all([c not in self.ood_validation_domains for c in self.train_domains])
        assert all([c not in self.ood_test_domains for c in self.train_domains])
        assert all([c not in self.ood_test_domains for c in self.ood_validation_domains])

        # buffers for datasets (accumulated when setup is called)
        self._train_datasets = list()
        self._id_validation_datasets = list()
        self._ood_validation_datasets = list()
        self._id_test_datasets = list()
        self._ood_test_datasets = list()

    def prepare_data(self) -> None:
        if not os.path.isdir(self.root):
            raise FileNotFoundError
        
    def setup(self, stage: str = None) -> None:
            
        for domain in track(self.train_domains, total=len(self.train_domains),
                            description='Preparing train / id-val / id-test data...'):
            
            # (1) train
            self._train_datasets += [
                IWildCam(self.root, locations=[domain], split='train', metadata=self.metadata)
            ]

            # (2) id-val
            if domain in self.id_validation_domains:
                self._id_validation_datasets += [
                    IWildCam(self.root, locations=[domain], split='id_val', metadata=self.metadata)
                ]

            # (3) id-test
            if domain in self.id_test_domains:
                self._id_test_datasets += [
                    IWildCam(self.root, locations=[domain], split='id_test', metadata=self.metadata)
                ]

        # (4) ood-val
        for domain in track(self.ood_validation_domains, total=len(self.ood_validation_domains),
                            description='Preparing ood-val data...'):
            self._ood_validation_datasets += [
                IWildCam(self.root, locations=[domain], split=None, metadata=self.metadata)
            ]
        
        # (5) ood-test
        for domain in track(self.ood_test_domains, total=len(self.ood_test_domains),
                            description='Preparing ood-test data...'):
            self._ood_test_datasets += [
                IWildCam(self.root, locations=[domain], split=None, metadata=self.metadata)
            ]

    def train_dataloader(self, infinite: bool = False) -> typing.Union[DataLoader, InfiniteDataLoader]:
        dataset = ConcatDataset(self._train_datasets)
        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
            shuffle=True,
        )
        if infinite:
            return InfiniteDataLoader(dataset, **loader_kwargs)
        else:
            return DataLoader(dataset, **loader_kwargs)
    
    def val_dataloader(self, data_type: str = 'ood', **kwargs) -> DataLoader:
        if data_type == 'id':
            return self._id_val_dataloader(**kwargs)
        elif data_type == 'ood':
            return self._ood_val_dataloader(**kwargs)
        else:
            raise ValueError

    def test_dataloader(self, data_type: str = 'ood', **kwargs) -> DataLoader:
        if data_type == 'id':
            return self._id_test_dataloader(**kwargs)
        elif data_type == 'ood':
            return self._ood_test_dataloader(**kwargs)
        else:
            raise ValueError

    def _id_val_dataloader(self, **kwargs) -> DataLoader:
        return self._get_dataloader_with_kwargs(
            ConcatDataset(self._id_validation_datasets), **kwargs
        )

    def _ood_val_dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(
            ConcatDataset(self._ood_validation_datasets), **kwargs
        )

    def _id_test_dataloader(self, **kwargs) -> DataLoader:
        return self._get_dataloader_with_kwargs(
            ConcatDataset(self._id_test_datasets), **kwargs
        )

    def _ood_test_dataloader(self, **kwargs) -> DataLoader:
        return self._get_dataloader_with_kwargs(
            ConcatDataset(self._ood_test_datasets), **kwargs
        )

    def _get_dataloader_with_kwargs(self, dataset: torch.utils.data.Dataset, **kwargs):
        return DataLoader(
            dataset,
            batch_size=kwargs.get('batch_size', self.batch_size),
            num_workers=kwargs.get('num_workers', self.num_workers),
            prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
        )

    @classmethod
    def get_domain_splits(cls, metadata: pd.DataFrame) -> typing.Dict[str, typing.List[int]]:

        train_mask = metadata['split'] == 'train'
        train_domains = sorted(metadata.loc[train_mask, cls._domain_col].unique().tolist())

        id_val_mask = metadata['split'] == 'id_val'
        id_val_domains = sorted(metadata.loc[id_val_mask, cls._domain_col].unique().tolist())
        
        ood_val_mask = metadata['split'] == 'val'
        ood_val_domains = sorted(metadata.loc[ood_val_mask, cls._domain_col].unique().tolist())

        id_test_mask = metadata['split'] == 'id_test'
        id_test_domains = sorted(metadata.loc[id_test_mask, cls._domain_col].unique().tolist())

        ood_test_mask = metadata['split'] == 'test'
        ood_test_domains = sorted(metadata.loc[ood_test_mask, cls._domain_col].unique().tolist())

        return {
            'train': train_domains,
            'id_val': id_val_domains,
            'ood_val': ood_val_domains,
            'id_test': id_test_domains,
            'ood_test': ood_test_domains,
        }


class SemiIWildCamDataModule(SemiSupervisedDataModule):
    def __init__(self,
                 root: str = 'data/wilds/iwildcam_v2.0',
                 unlabeled_root: str = 'data/wilds/iwildcam_unlabeled_v1.0',
                 batch_size: typing.Optional[int] = 16,
                 num_workers: typing.Optional[int] = 8,
                 prefetch_factor: typing.Optional[int] = 4,
                 ):
        
        super().__init__()

        # dataset arguments
        self.root = root
        self.unlabeled_root = unlabeled_root

        # dataloader arguments
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        # read metadata (labeled)
        self.metadata = pd.read_csv(os.path.join(self.root, 'metadata.csv'), index_col=0)

        # dictionary of list of domains
        domain_splits = IWildCamDataModule.get_domain_splits(self.metadata)
        
        self.train_domains = domain_splits['train']
        self.id_validation_domains = domain_splits['id_val']
        self.ood_validation_domains = domain_splits['ood_val']
        self.id_test_domains = domain_splits['id_test']
        self.ood_test_domains = domain_splits['ood_test']

        # read metadata (unlabeled)
        self.metadata_u = pd.read_csv(os.path.join(self.unlabeled_root, 'metadata.csv'), index_col=0)
        self.unlabeled_domains = sorted(self.metadata_u['location_remapped'].unique().tolist())

        self._labeled_train_datasets = list()
        self._unlabeled_train_datasets = list()
        self._id_validation_datasets = list()
        self._ood_validation_datasets = list()
        self._id_test_datasets = list()
        self._ood_test_datasets = list()

    def prepare_data(self) -> None:
        if not os.path.isdir(self.root):
            raise FileNotFoundError
        if not os.path.isdir(self.unlabeled_root):
            raise FileNotFoundError

    def setup(self, stage: str = None) -> None:

        for domain in track(self.train_domains, total=len(self.train_domains),
                            description='Preparing train / id-val / id-test data...'):
            
            # (1) train
            self._labeled_train_datasets += [
                IWildCam(self.root, locations=[domain], split='train', metadata=self.metadata)
            ]

            # (2) id-val
            if domain in self.id_validation_domains:
                self._id_validation_datasets += [
                    IWildCam(self.root, locations=[domain], split='id_val', metadata=self.metadata)
                ]

            # (3) id-test
            if domain in self.id_test_domains:
                self._id_test_datasets += [
                    IWildCam(self.root, locations=[domain], split='id_test', metadata=self.metadata)
                ]
                
        # (4) ood-val
        for domain in track(self.ood_validation_domains, total=len(self.ood_validation_domains),
                            description='Preparing ood-val data...'):
            self._ood_validation_datasets += [
                IWildCam(self.root, locations=[domain], split=None, metadata=self.metadata)
            ]

        # (5) ood-test
        for domain in track(self.ood_test_domains, total=len(self.ood_test_domains),
                            description='Preparing ood-test data...'):
            self._ood_test_datasets += [
                IWildCam(self.root, locations=[domain], split=None, metadata=self.metadata)
            ]  

        # (6) unlabeled extra
        for domain in track(self.unlabeled_domains, total=len(self.unlabeled_domains),
                            description='Preparing unlabeled training data...'):
            self._unlabeled_train_datasets += [
                UnlabeledIWildCam(self.unlabeled_root, locations=[domain], metadata=self.metadata_u)
            ]

    def train_dataloader(self, infinite: bool = False, **kwargs):
        return self._labeled_dataloader(infinite, **kwargs), self._unlabeled_dataloader(infinite, **kwargs)

    def _labeled_dataloader(self, infinite: bool = False, **kwargs):
        dataset = ConcatDataset(self._labeled_train_datasets)
        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
            shuffle=True,
        )
        if infinite:
            return InfiniteDataLoader(dataset, **loader_kwargs)
        else:
            return DataLoader(dataset, **loader_kwargs)
    
    def _unlabeled_dataloader(self, infinite: bool = False, **kwargs):
        dataset = ConcatDataset(self._unlabeled_train_datasets)
        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
            shuffle=True,
        )
        if infinite:
            return InfiniteDataLoader(dataset, **loader_kwargs)
        else:
            return DataLoader(dataset, **loader_kwargs)
    
    def val_dataloader(self, data_type: str = 'ood', **kwargs):
        if data_type == 'ood':
            return self._ood_val_dataloader(**kwargs)
        elif data_type == 'id':
            return self._id_val_dataloader(**kwargs)
        else:
            raise ValueError
        
    def test_dataloader(self, data_type: str = 'ood', **kwargs):
        if data_type == 'ood':
            return self._ood_test_dataloader(**kwargs)
        elif data_type == 'id':
            return self._id_test_dataloader(**kwargs)
        else:
            raise ValueError
    
    def _id_val_dataloader(self, **kwargs) -> DataLoader:
        return self._get_dataloader_with_kwargs(
            ConcatDataset(self._id_validation_datasets), **kwargs
        )

    def _ood_val_dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(
            ConcatDataset(self._ood_validation_datasets), **kwargs
        )

    def _id_test_dataloader(self, **kwargs) -> DataLoader:
        return self._get_dataloader_with_kwargs(
            ConcatDataset(self._id_test_datasets), **kwargs
        )

    def _ood_test_dataloader(self, **kwargs) -> DataLoader:
        return self._get_dataloader_with_kwargs(
            ConcatDataset(self._ood_test_datasets), **kwargs
        )

    def _get_dataloader_with_kwargs(self, dataset: torch.utils.data.Dataset, **kwargs):
        return DataLoader(
            dataset,
            batch_size=kwargs.get('batch_size', self.batch_size),
            num_workers=kwargs.get('num_workers', self.num_workers),
            prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
        )

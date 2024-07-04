
import os
import time
import typing
import functools

import torch
import numpy as np
import pandas as pd
import lightning as L

from ray.util.multiprocessing import Pool as RayPool
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from dg_lightning.datasets.base import MultipleDomainDataModule
from dg_lightning.datasets.loaders import InfiniteDataLoader


class Camelyon17(Dataset):

    _allowed_hospitals = (0, 1, 2, 3, 4)

    def __init__(self,
                 root: str = 'data/wilds/camelyon17_v1.0',
                 hospitals: typing.Iterable[int] = [0],
                 split: typing.Optional[str] = None,
                 in_memory: typing.Optional[int] = 0,
                 ) -> None:
        super().__init__()

        self.root = root
        self.hospitals = hospitals
        self.split = split
        self.in_memory = in_memory

        for h in self.hospitals:
            if h not in self._allowed_hospitals:
                raise ValueError
        
        if self.split is not None:
            if self.split not in ('train', 'val'):
                raise ValueError
            
        # Read metadata
        metadata = pd.read_csv(
            os.path.join(self.root, 'metadata.csv'),
            index_col=0,
            dtype={'patient': 'str'}
        )

        # Keep rows of metadata specific to hospital(s) & split
        rows_to_keep = metadata['center'].isin(hospitals)
        if self.split == 'train':
            rows_to_keep = rows_to_keep & (metadata['split'] == 0)
        elif self.split == 'val':
            rows_to_keep = rows_to_keep & (metadata['split'] == 1)

        metadata = metadata.loc[rows_to_keep].copy()
        metadata = metadata.reset_index(drop=True, inplace=False)

        # Main attributes
        self.input_files = [
            os.path.join(
                self.root,
                f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
            ) for patient, node, x, y in
            metadata.loc[:, ['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False, name=None)
        ]
        if self.in_memory > 0:
            start = time.time()
            print(f'Loading {len(self.input_files):,} images in memory (hospital={self.hospital}, split={self.split}).', end=' ')
            self.inputs = self.load_images(self.input_files, p=self.in_memory, as_tensor=True)
            print(f'Elapsed Time: {time.time() - start:.2f} seconds.')
        else:
            self.inputs = None
        
        self.targets = torch.LongTensor(metadata['tumor'].values)
        self.domains = torch.LongTensor(metadata['center'].values)
        self.eval_groups = torch.LongTensor(metadata['slide'].values)
        self.metadata = metadata

    def get_input(self, index: int) -> torch.ByteTensor:
        if self.input is not None:
            return self.inputs[index]
        else:
            return read_image(self.input_files[index], mode=ImageReadMode.RGB)
        
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
    def load_images(filenames: typing.List[str],
                    p: int,
                    as_tensor: bool = True,
                    ) -> typing.Union[typing.List[torch.Tensor], torch.Tensor]:
        """
        Load images with multiprocessing if p > 0.
        Arguments:
            filenames: list of filename strings.
            p: int for number of cpu threads to use for data loading.
            as_tensor: bool, returns a stacked tensor if True, a list of tensor images if False.
        Returns:
            ...
        """
        with RayPool(processes=p) as pool:
            images = pool.map(functools.partial(read_image, mode=ImageReadMode.RGB), filenames)
            pool.close(); pool.join(); time.sleep(5.0)

        return torch.stack(images, dim=0) if as_tensor else images

    @staticmethod
    def download(root: str) -> None:
        raise NotImplementedError
    
    @property
    def input_shape(self) -> typing.Tuple[int]:
        return (3, 96, 96)
    
    @property
    def num_classes(self) -> int:
        return 2


class UnlabeledCamelyon17(Dataset):

    _allowed_hospitals = (0, 1, 2, 3, 4)

    def __init__(self,
                 root: str = 'data/wilds/camelyon17_unlabeled_v1.0',
                 hospitals: typing.Iterable[int] = [0],
                 ) -> None:
        
        super().__init__()

        self.root = root
        self.hospitals = hospitals

        for h in self.hospitals:
            if h not in self._allowed_hospitals:
                raise ValueError
            
        metadata = pd.read_csv(
            os.path.join(self.root, 'metadata.csv'),
            index_col=0,
            dtype={'patient': 'str'},
        )

        # keep rows of metadata specific to hospital(s) & split
        rows_to_keep = metadata['center'].isin(hospitals)
        metadata = metadata.loc[rows_to_keep].copy()
        metadata = metadata.reset_index(drop=True, inplace=False)

        self.input_files = [
            os.path.join(
                self.root,
                f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
            ) for patient, node, x, y in
            metadata.loc[:, ["patient", "node", "x_coord", "y_coord"]].itertuples(index=False, name=None)
        ]

        self.domains = torch.LongTensor(metadata['center'].values)
        self.eval_groups = torch.LongTensor(metadata['slide'].values)
        self.metadata = metadata

    def get_input(self, index: int) -> torch.ByteTensor:
        if self.input is not None:
            return self.inputs[index]
        else:
            return read_image(self.input_files[index], mode=ImageReadMode.RGB)
        
    def get_domain(self, index: int) -> torch.LongTensor:
        return self.domains[index]
    
    def get_eval_group(self, index: int) -> torch.LongTensor:
        return self.eval_groups[index]
    
    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return dict(
            x=self.get_input(index),
            domain=self.get_domain(index),
            eval_group=self.get_eval_group(index),
        )
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    @staticmethod
    def download(root: str) -> None:
        raise NotImplementedError


class Camelyon17DataModule(MultipleDomainDataModule):
    def __init__(self,
                 root: str = './data/wilds/camelyon17_v1.0',
                 train_domains: typing.Iterable[int] = [0, 3, 4],
                 validation_domains: typing.Iterable[int] = [1],
                 test_domains: typing.Iterable[int] = [2],
                 batch_size: typing.Optional[int] = 32,
                 num_workers: typing.Optional[int] = 4,
                 prefetch_factor: typing.Optional[int] = 2,
                 ) -> None:
        
        super().__init__()
        
        # dataset arguments
        self.root = root
        self.train_domains = [int(d) for d in train_domains]
        self.validation_domains = [int(d) for d in validation_domains]
        self.test_domains = [int(d) for d in test_domains]
        
        # dataloader arguments
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        # buffers for datasets (accumulated when setup is called)
        self._train_datasets = list()
        self._id_validation_datasets = list()
        self._ood_validation_datasets = list()
        self._id_test_datasets = list()
        self._ood_test_datasets = list()

    def prepare_data(self) -> None:
        if not os.path.isdir(self.root):
            raise FileNotFoundError("Download data using `download_wilds_data.py`.")

    def setup(self, stage: typing.Optional[str] = None) -> None:
        
        if stage in (None, 'fit'):
            
            for domain in self.train_domains:
                self._train_datasets += [
                    Camelyon17(self.root, hospitals=[domain], split='train')
                ]

        if stage in (None, 'fit', 'validate'):

            for domain in self.train_domains:
                self._id_validation_datasets += [
                    Camelyon17(self.root, hospitals=[domain], split='val')
                ]

            for domain in self.validation_domains:
                self._ood_validation_datasets += [
                    Camelyon17(self.root, hospitals=[domain], split=None)
                ]
        
        if stage in (None, 'test'):            
            
            for domain in self.test_domains:
                self._ood_test_datasets += [
                    Camelyon17(self.root, hospitals=[domain], split=None)
                ]

            for domain in self.train_domains:
                pass  # TODO: add in-distribution test data

    def train_dataloader(self, infinite: bool = False) -> typing.Union[DataLoader, InfiniteDataLoader]:
        dataset = ConcatDataset(self._train_datasets)
        if infinite:
            return self._infinite_dataloader(dataset, shuffle=True, drop_last=True)
        else:
            return self._finite_dataloader(dataset, shuffle=True, drop_last=True)
        
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
        return self._finite_dataloader(ConcatDataset(self._id_validation_datasets), **kwargs)

    def _ood_val_dataloader(self, **kwargs) -> DataLoader:
        return self._finite_dataloader(ConcatDataset(self._ood_validation_datasets), **kwargs)

    def _id_test_dataloader(self, **kwargs) -> DataLoader:
        return self._finite_dataloader(ConcatDataset(self._id_test_datasets), **kwargs)

    def _ood_test_dataloader(self, **kwargs) -> DataLoader:
        return self._finite_dataloader(ConcatDataset(self._ood_test_datasets), **kwargs)
    
    def _finite_dataloader(self, dataset: Dataset, **kwargs) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            shuffle=kwargs.get('shuffle', True),
            drop_last=kwargs.get('drop_last', True),
            batch_size=kwargs.get('batch_size', self.batch_size),
            num_workers=kwargs.get('num_workers', self.num_workers),
            prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
            pin_memory=True,
        )

    def _infinite_dataloader(self, dataset: Dataset, **kwargs) -> InfiniteDataLoader:
        return InfiniteDataLoader(
            dataset=dataset,
            shuffle=kwargs.get('shuffle', True),
            drop_last=kwargs.get('drop_last', True),
            batch_size=kwargs.get('batch_size', self.batch_size),
            num_workers=kwargs.get('num_workers', self.num_workers),
            prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
            pin_memory=True,
        )


class SemiCamelyon17DataModule(Camelyon17DataModule):
    def __init__(self,
                 root: str = 'data/wilds/camelyon17_v1.0',
                 unlabeled_root: str = 'data/wilds/camelyon17_unlabeled_v1.0',
                 train_domains: typing.Iterable[int] = [0, 3, 4],
                 validation_domains: typing.Iterable[int] = [1],
                 test_domains: typing.Iterable[int] = [2],
                 batch_size: typing.Optional[int] = 32,
                 num_workers: typing.Optional[int] = 4,
                 prefetch_factor: typing.Optional[int] = 2,
                 ):
        
        super().__init__(
            root=root,
            train_domains=train_domains,
            validation_domains=validation_domains,
            test_domains=test_domains,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

        self.unlabeled_root: str = unlabeled_root
        self._unlabeled_datasets: typing.List[Dataset] = list()

    def prepare_data(self) -> None:
        if not (os.path.isdir(self.root) and os.path.isdir(self.unlabeled_root)):
            raise FileNotFoundError("Download data using `download_wilds_data.py`.")
            
    def setup(self, stage: typing.Optional[str] = None) -> None:

        if stage in (None, 'fit'):
            for domain in self.train_domains:
                self._train_datasets += [
                    Camelyon17(self.root, hospitals=[domain], split='train')
                ]
                self._unlabeled_datasets += [
                    UnlabeledCamelyon17(self.unlabeled_root, hospitals=[domain], split=None)
                ]

        if stage in (None, 'fit', 'validate'):
            for domain in self.train_domains:
                self._id_validation_datasets += [
                    Camelyon17(self.root, hospitals=[domain], split='val')
                ]
            for domain in self.validation_domains:
                self._ood_validation_datasets += [
                    Camelyon17(self.root, hospitals=[domain], split=None)
                ]

        if stage in (None, 'test'):
            for domain in self.test_domains:
                self._ood_test_datasets += [
                    Camelyon17(self.root, hospitals=[domain], split=None)
                ]

    def train_dataloader(self, infinite: bool = False, **kwargs):
        return self._labeled_dataloader(infinite, **kwargs), self._unlabeled_dataloader(infinite, **kwargs)

    def _labeled_dataloader(self, infinite: bool = False, **kwargs):
        dataset = ConcatDataset(self._train_datasets)
        if infinite:
            return self._infinite_dataloader(dataset, shuffle=True, drop_last=True)
        else:
            return self._finite_dataloader(dataset, shuffle=True, drop_last=True)
    
    def _unlabeled_dataloader(self, infinite: bool = False, **kwargs):
        dataset = ConcatDataset(self._unlabeled_datasets)
        if infinite:
            return self._infinite_dataloader(dataset, shuffle=True, drop_last=True)
        else:
            return self._finite_dataloader(dataset, shuffle=True, drop_last=True)
    
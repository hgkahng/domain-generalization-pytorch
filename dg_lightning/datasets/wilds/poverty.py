
import os
import time
import typing
import inspect
import functools
import argparse

import torch
import numpy as np
import pandas as pd
import lightning as L

from torch.utils.data import DataLoader, ConcatDataset


def subsample_idxs(idxs: np.ndarray, num: int = 5000, take_rest: bool = False, seed= None) -> np.ndarray:
    """
    Reference:
        https://github.com/p-lambda/wilds/blob/472677590de351857197a9bf24958838c39c272b/wilds/common/utils.py#L104
    """
    seed = (seed + 541433) if seed is not None else None
    rng = np.random.default_rng(seed)

    idxs = idxs.copy()
    rng.shuffle(idxs)
    if take_rest:
        idxs = idxs[num:]
    else:
        idxs = idxs[:num]
    return idxs


class SinglePovertyMap(torch.utils.data.Dataset):
    
    _allowed_countries = [
        'angola', 'benin', 'burkina_faso', 'cameroon', 'cote_d_ivoire',
        'democratic_republic_of_congo', 'ethiopia', 'ghana', 'guinea', 'kenya',
        'lesotho', 'malawi', 'mali', 'mozambique', 'nigeria', 'rwanda', 'senegal',
        'sierra_leone', 'tanzania', 'togo', 'uganda', 'zambia', 'zimbabwe'
    ]  # 23
    
    _allowed_areas = ['urban', 'rural']

    _BAND_ORDER = [
        'BLUE', 'GREEN', 'RED', 'SWIR1',
        'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS'
    ]

    _SURVEY_NAMES_2009_17A = {  # fold = A
        'train': ['cameroon', 'democratic_republic_of_congo', 'ghana', 'kenya',
                  'lesotho', 'malawi', 'mozambique', 'nigeria', 'senegal',
                  'togo', 'uganda', 'zambia', 'zimbabwe'],
        'ood_val': ['benin', 'burkina_faso', 'guinea', 'sierra_leone', 'tanzania'],
        'ood_test': ['angola', 'cote_d_ivoire', 'ethiopia', 'mali', 'rwanda'],
    }

    _SURVEY_NAMES_2009_17B = {  # fold = B
        'train': ['angola', 'cote_d_ivoire', 'democratic_republic_of_congo',
                  'ethiopia', 'kenya', 'lesotho', 'mali', 'mozambique',
                  'nigeria', 'rwanda', 'senegal', 'togo', 'uganda', 'zambia'],
        'ood_val': ['cameroon', 'ghana', 'malawi', 'zimbabwe'],
        'ood_test': ['benin', 'burkina_faso', 'guinea', 'sierra_leone', 'tanzania'],
    }

    _SURVEY_NAMES_2009_17C = {  # fold = C
        'train': ['angola', 'benin', 'burkina_faso', 'cote_d_ivoire', 'ethiopia',
                  'guinea', 'kenya', 'lesotho', 'mali', 'rwanda', 'senegal',
                  'sierra_leone', 'tanzania', 'zambia'],
        'ood_val': ['democratic_republic_of_congo', 'mozambique', 'nigeria', 'togo', 'uganda'],
        'ood_test': ['cameroon', 'ghana', 'malawi', 'zimbabwe'],
    }

    _SURVEY_NAMES_2009_17D = {  # fold = D
        'train': ['angola', 'benin', 'burkina_faso', 'cameroon', 'cote_d_ivoire',
                  'ethiopia', 'ghana', 'guinea', 'malawi', 'mali', 'rwanda',
                  'sierra_leone', 'tanzania', 'zimbabwe'],
        'ood_val': ['kenya', 'lesotho', 'senegal', 'zambia'],
        'ood_test': ['democratic_republic_of_congo', 'mozambique', 'nigeria', 'togo', 'uganda'],
    }

    _SURVEY_NAMES_2009_17E = {  # fold = E
        'train': ['benin', 'burkina_faso', 'cameroon', 'democratic_republic_of_congo',
                  'ghana', 'guinea', 'malawi', 'mozambique', 'nigeria', 'sierra_leone',
                  'tanzania', 'togo', 'uganda', 'zimbabwe'],
        'ood_val': ['angola', 'cote_d_ivoire', 'ethiopia', 'mali', 'rwanda'],
        'ood_test': ['kenya', 'lesotho', 'senegal', 'zambia'],
    }

    def __init__(self,
                 root: str = 'data/wilds/poverty_v1.1',
                 country: typing.Union[str, int] = 'rwanda',  # 16
                 split: str = 'train',                        # train, id_val, id_test, ood_val, ood_test
                 random_state: int = 42,
                 in_memory: int = 0,
                 ) -> None:

        self.root = root; assert os.path.isdir(self.root)
        self.split = split; assert self.split in ['train', 'id_val', 'id_test', 'ood_val', 'ood_test', None]
        if isinstance(country, int):
            self.country: str = self._allowed_countries[country]
        else:
            self.country: str = country
        assert self.country in self._allowed_countries
        self.random_state = random_state
        self.in_memory = in_memory

        # TODO: remove if unused
        self._country2idx = {c: i for i, c in enumerate(self._allowed_countries)}
        self._idx2country = self._allowed_countries

        # load metadata
        metadata = pd.read_csv(os.path.join(self.root, 'dhs_metadata.csv'))
        metadata['area'] = metadata['urban'].apply(lambda b: 'urban' if b else 'rural')
        metadata['original_idx'] = metadata.index.values  # IMPORTANT; for input file configuration

        # keeps rows specific to country
        country_indices = np.where(metadata['country'] == self.country)[0]

        # keep rows specific to split
        if self.split not in ['train', 'id_val', 'id_test']:
            indices = country_indices
        else:
            N = int(len(country_indices) * 0.2)  # FIXME: use split provided by original wilds
            if self.split == 'train':
                indices = subsample_idxs(country_indices, take_rest=True, num=N, seed=random_state)
            elif self.split == 'id_val':
                indices = subsample_idxs(country_indices, take_rest=False, num=N, seed=random_state)
                indices = indices[N//2:]
            elif self.split == 'id_test':
                indices = subsample_idxs(country_indices, take_rest=False, num=N, seed=random_state)
                indices = indices[:N//2]
            else:
                indices = country_indices
        
        self.metadata = metadata.iloc[indices].reset_index(drop=True, inplace=False)

        # TODO: remove if unused
        self._true_indices = indices

        # list of input files (IMPORTANT)
        self.input_files: typing.List[str] = [
            os.path.join(
                self.root, f'images/landsat_poverty_img_{idx}.npz'
            ) for idx in self.metadata['original_idx'].values
        ]
        assert all([os.path.exists(f) for f in self.input_files])
        if self.in_memory > 0:
            raise NotImplementedError
        else:
            self.inputs = None

        # targets, domains, evaluation groups (IMPORTANT)
        self.targets = torch.from_numpy(self.metadata['wealthpooled'].values).float()
        self.domains = torch.tensor(
            [self._allowed_countries.index(c) for c in self.metadata['country'].values],
            dtype=torch.long,
        )
        self.eval_groups = torch.from_numpy(self.metadata['urban'].astype(int).values).long()  # 1 for urban, 0 for rural
        
    def get_input(self, index: int) -> torch.FloatTensor:
        if self.inputs is not None:
            return self.inputs[index]
        else:
            img: np.ndarray = np.load(self.input_files[index])['x']  # already np.float32
            return torch.from_numpy(img).float()

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


class PovertyMapDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str = './data/wilds/poverty_v1.1',
        train_domains: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17A['train'],
        validation_domains: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17A['ood_val'],
        test_domains: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17A['ood_test'],
        batch_size: int = 32,
        num_workers: int = 4,
        prefetch_factor: typing.Optional[int] = 2,
        ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # dataset arguments
        self.root = root

        # dataloader arguments
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        # list of domain strings
        self._train_countries = train_domains
        self._validation_countries = validation_domains
        self._test_countries = test_domains
        
        # list of domain integers
        self.train_domains = [SinglePovertyMap._allowed_countries.index(c) for c in self._train_countries]
        self.validation_domains = [SinglePovertyMap._allowed_countries.index(c) for c in self._validation_countries]
        self.test_domains = [SinglePovertyMap._allowed_countries.index(c) for c in self._test_countries]

    def prepare_data(self) -> None:
        if not os.path.isdir(self.root):
            raise FileNotFoundError

    def setup(self, stage: typing.Optional[str] = None) -> None:

        # collection of datasets
        self._train_datasets = list()
        self._id_validation_datasets = list()
        self._ood_validation_datasets = list()
        self._id_test_datasets = list()
        self._ood_test_datasets = list()

        # TODO: change 'random_state' based on fold

        if (stage is None) or (stage == 'fit') or (stage == 'validate'):

            # (1) train / id-validation
            for c in self._train_countries:
                self._train_datasets += [
                    SinglePovertyMap(root=self.root, country=c, split='train', random_state=42)
                ]
                self._id_validation_datasets += [
                    SinglePovertyMap(root=self.root, country=c, split='id_val', random_state=42)
                ]

            # (2) ood-validation
            for c in self._validation_countries:
                self._ood_validation_datasets += [
                    SinglePovertyMap(root=self.root, country=c, split='ood_val', random_state=42)
                ]

        if (stage is None) or (stage == 'test'):

            # (3) ood-test
            for c in self._test_countries:
                self._ood_test_datasets += [
                    SinglePovertyMap(root=self.root, country=c, split='ood_test', random_state=42)
                ]

            # (4) id-test
            for c in self._train_countries:
                self._id_test_datasets += [
                    SinglePovertyMap(root=self.root, country=c, split='id_test', random_state=42)
                ]

    def train_dataloader(self, sampler = None) -> DataLoader:
        return DataLoader(
            dataset=ConcatDataset(self._train_datasets),
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=True,
            **self.general_loader_config
        )

    def val_dataloader(self) -> DataLoader:
        if len(self._ood_validation_datasets) > 0:
            return self._ood_val_dataloader()
        else:
            return self._id_val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return self._ood_test_dataloader()

    def _id_val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=ConcatDataset(self._id_validation_datasets),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            **self.general_loader_config
        )

    def _ood_val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=ConcatDataset(self._ood_validation_datasets),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            **self.general_loader_config
        )

    def _id_test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=ConcatDataset(self._id_test_datasets),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            **self.general_loader_config
        )

    def _ood_test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=ConcatDataset(self._ood_test_datasets),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            **self.general_loader_config
        )

    @property
    def general_loader_config(self) -> typing.Dict[str, typing.Any]:
        return {
            'num_workers': self.num_workers,
            'prefetch_factor': self.prefetch_factor,
        }

    @classmethod
    def from_argparse_args(cls, args: argparse.Namespace) -> L.LightningDataModule:
        init_arg_names = [k for k in inspect.signature(cls.__init__).parameters]
        init_kwargs = {k: v for k, v in vars(args).items() if k in init_arg_names}
        return cls(**init_kwargs)

    @classmethod
    def add_data_specific_args(cls,
                               parent_parser: argparse.ArgumentParser,
                               ) -> argparse.ArgumentParser:
        
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        
        group = parser.add_argument_group(f"{cls.__name__}")
        group.add_argument('--fold', type=str, choices=('A', 'B', 'C', 'D', 'E'), help='')
        group.add_argument('--batch_size', type=int, default=16, help='')
        group.add_argument('--num_workers', type=int, default=4, help='')
        group.add_argument('--pin_memory', dest='pin_memory', action='store_true')
        group.add_argument('--prefetch_factor', type=int, default=2, help='')
        group.add_argument('--persistent_workers', dest='persistent_workers', action='store_true')
        
        return parser
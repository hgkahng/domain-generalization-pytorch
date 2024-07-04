
import inspect
import argparse

import lightning as L


class MultipleDomainDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
    
    def prepare_data(self):
        """
        Code for downloading data.
            # download, split, etc...
            # only called on 1 GPU/TPU in distributed (on main process)
        """
        raise NotImplementedError

    def setup(self, stage: str):
        """
        Code for assigning train/val/test datasets for use in dataloaders.
        `setup()` is called after `prepare_data()` and there is a barrier in
        between which ensures that all the processes proceed to setup once the
        data is prepared and available for use.
            # make assignments here (val/train/test split)
            # called on every process in DDP
        """
        raise NotImplementedError
    
    def train_dataloader(self):
        raise NotImplementedError
    
    def val_dataloader(self):
        raise NotImplementedError
    
    def test_dataloader(self):
        raise NotImplementedError
    
    def predict_dataloader(self):
        return NotImplementedError
    
    @classmethod
    def from_argparse_args(cls,
                           args: argparse.Namespace,
                           ) -> L.LightningDataModule:
        init_arg_names = [k for k in inspect.signature(cls.__init__).parameters]
        init_kwargs = {k: v for k, v in vars(args).items() if k in init_arg_names}
        return cls(**init_kwargs)


class SupervisedDataModule(MultipleDomainDataModule):
    def __init__(self):
        super().__init__()


class SemiSupervisedDataModule(MultipleDomainDataModule):
    def __init__(self):
        super().__init__()


class SelfSupervisedDataModule(MultipleDomainDataModule):
    def __init__(self):
        super().__init__()

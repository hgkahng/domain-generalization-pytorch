
import inspect
import argparse

from typing import Union, Dict, Any

import lightning as L


def from_argparse_args(cls,
                       args: Union[argparse.Namespace, Dict[str, Any]],
                       **kwargs) -> Union[L.LightningModule, L.LightningDataModule]:
    init_arg_names = [k for k in inspect.signature(cls.__init__).parameters]
    init_kwargs = {k: v for k, v in vars(args).items() if k in init_arg_names}
    if kwargs:
        init_kwargs.update(kwargs)
    return cls(**init_kwargs)

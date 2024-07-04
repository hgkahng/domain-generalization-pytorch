
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )
)  # appending project folder to list of system paths

import yaml
import warnings
import argparse

import torch
import lightning as L

from rich.console import Console
from rich.table import Table

from dg_lightning.models.erm import EmpiricalRiskMinimization
from dg_lightning.datasets.wilds.camelyon17 import Camelyon17DataModule


data2metric = {
    'camelyon17': ('accuracy', 'max'),
    'pacs': ('accuracy', 'max'),  # TODO:
}


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(prog="ERM", add_help=True)
    parser = EmpiricalRiskMinimization.add_model_specific_args(parser)  # TODO: 
    parser = Camelyon17DataModule.add_data_specific_args(parser)        # TODO: 

    # trainer arguments
    parser.add_argument('--gpus', nargs='+', type=int, default=[], help='')
    parser.add_argument('--max_epochs', type=int, default=5, help='')

    # misc arguments
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='')
    parser.add_argument('--hash', type=str, default=None, help='')
    
    # override argument defaults from yaml config files
    parser.add_argument('--config', action='append')
    args, _ = parser.parse_known_args()
    if args.config is not None:
        assert isinstance(args.config, list)
        for conf_fname in args.config:
            assert conf_fname.endswith('.yaml')
            with open(conf_fname, 'r') as f:
                cfg = yaml.safe_load(f)
                cfg = {k: v for k, v in cfg.items() if k in vars(args)}
                parser.set_defaults(**cfg)  # override

    args, _ = parser.parse_known_args()
    if args.hash is None:
        import datetime
        setattr(args, 'hash', datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    
    return args


def main(args: argparse.Namespace) -> None:
    
    L.seed_everything(args.seed, workers=True)

    # (0) print configurations
    console = Console()
    table = Table(title='Configuration (ERM)')
    table.add_column('Name', justify='right', style='white')
    table.add_column('Value', justify='left', style='green')
    _ = [table.add_row(str(k), str(v)) for k, v in vars(args).items()]
    console.print(table);
    
    # (2) data module
    if args.data == 'camelyon17':
        dm = Camelyon17DataModule.from_argparse_args(args)
    elif args.data == 'pacs':
        raise NotImplementedError  # TODO:
    else:
        raise ValueError

    # (3) training module
    model = EmpiricalRiskMinimization.from_argparse_args(args)
    trainer = L.Trainer(
        accelerator='gpu' if len(args.gpus) > 0 else 'cpu',
        devices=args.gpus,
        max_epochs=args.max_epochs,
        num_nodes=1,
        precision='16-mixed',  # FIXME;
        logger=[
            L.loggers.CSVLogger(
                save_dir=os.path.join(
                    args.checkpoint_dir,
                    f"{args.data}",
                    f"{model.__class__.__name__}"),
                name=args.hash,
                version=0,
            )
        ],
        callbacks=[
            L.callbacks.RichProgressBar(leave=True),
            L.callbacks.RichModelSummary(max_depth=1),
            L.callbacks.LearningRateMonitor(logging_interval='epoch'),
            L.callbacks.ModelCheckpoint(
                dirpath=os.path.join(args.checkpoint_dir, f"{model.__class__.__name__}", args.hash),
                monitor=f'val_{data2metric[args.data][0]}',
                mode=f'{data2metric[args.data][-1]}',
            ),
        ],
        deterministic=True,  # full reproducibility
    )

    # fit
    trainer.fit(
        model=model,
        datamodule=dm,
    )

    # test
    trainer.test(
        model=model,
        ckpt_path='best',
        datamodule=dm,
    )


if __name__ == "__main__":

    warnings.filterwarnings(action='ignore')
    args = parse_arguments()
    try:
        main(args)
    except KeyboardInterrupt as _:
        sys.exit(0)

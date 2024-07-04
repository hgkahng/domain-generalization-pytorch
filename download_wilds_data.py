
"""
    Script used to download WILDS data.
    Reference:
        https://github.com/p-lambda/wilds/blob/472677590de351857197a9bf24958838c39c272b/wilds/download_datasets.py

    Example:
        # Downloading the labeled Camelyon17 dataset (target directory: ./data/wilds/camelyon17_v1.0)
        >> python download_wilds_data.py --root_dir ./data/wilds --datasets camelyon17
        
        # Downloading the unlabeled Camelyon17 dataset (target directory: ./data/wilds/camelyon17_unlabeled_v1.0)
        >> python download_wilds_data.py --root_dir ./data/wilds --datasets camelyon17 --unlabeled
"""

import os
import argparse
import wilds


SUPPORTED_WILDS_DATASETS = [
    'camelyon17',
    'poverty',
    'iwildcam',
    'rxrx1',
    'civilcomments',
]
assert all([s in wilds.supported_datasets for s in SUPPORTED_WILDS_DATASETS])


def parse_arguments():

    parser = argparse.ArgumentParser(prog="Download data.")
    parser.add_argument('--root_dir', required=True,
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')
    parser.add_argument('--datasets', nargs='*', default=SUPPORTED_WILDS_DATASETS,
                        choices=SUPPORTED_WILDS_DATASETS,
                        help=f'Specify a space-separated list of dataset names to download. If left unspecified, all datasets in {SUPPORTED_WILDS_DATASETS} will be downloaded.')
    parser.add_argument('--unlabeled', action='store_true',
                        help=f'If this flag is set, the unlabeled dataset will be downloaded instead of the labeled.')
    
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:

    # create root data directory if it does not exist
    os.makedirs(args.root_dir, exist_ok=True)

    print(f'Downloading the following datasets: {args.datasets}')
    for dataset in args.datasets:
        print(f'=== {dataset} ===')
        wilds.get_dataset(
            dataset=dataset,
            root_dir=args.root_dir,
            unlabeled=args.unlabeled,
            download=True
        )


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

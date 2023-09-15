import comet_ml
import yaml
import torch
import os
import argparse

from lightning import pytorch as pl
from models.lightning_wrapper import PyramidViGLT, ResNetLT
from models.vit_lightning import ViTLT
from torchgeo.datasets import RESISC45, PatternNet, BigEarthNet
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CometLogger
from pathlib import Path
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from functools import partial
from typing import Dict, List, Tuple, Optional, Union

def parse():
    parser = argparse.ArgumentParser(description='Train a model on BigEarthNet/RESISC45/PatternNet')
    parser.add_argument('-n', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch', type=int, default=2, help='Batch size')
    parser.add_argument('--model-config', type=str, default=None, help='Model config path (yaml file)')
    parser.add_argument('--comet-logger', dest='comet', default=False, action='store_true', help='Enable CometML logging')
    parser.add_argument('--dataset', type=str, help='Dataset used for training. Must be either {"patternnet", "resisc45", "bigearthnet"}')
    parser.add_argument('--split', type=str, default=None, help='Dataset split, valid only for patternnet dataset. Must be in format x,y,z, e.g. 70,15,15')
    parser.add_argument('--seed', type=int, default=47, help='Random seed. Default is 47.')
    parser.add_argument('--split-seed', type=int, default=None, help='Random seed. Default is None.')
    parser.add_argument('--pretrained-checkpoint', type=str, default=None, help='Path to pretrained checkpoint from which encoder weights are imported. If none, simple supervised training is done')
    parser.add_argument('--resume', type=str, default=None, help='Resume training. Path to ckpt file must be specified')
    parser.add_argument('--experiment-key', type=str, default=None, help='Experiment key for cometML logging')
    parser.add_argument('--checkpoint-folder', type=str, help='Checkpoint folder used to log this run')
    parser.add_argument('--exp-name', type=str, help='Experiment name postfix for comet')
    parser.add_argument('--debug', action='store_true', default=False, help='Fast debug run mode. Default is False')
    parser.add_argument('--test', type=str, default=None, help='If specified, the model checkpoints specified is loaded and a simple test evaluation is performed.')
    parser.add_argument('--val-batch', type=int, default=None, help='Validation batch size. If None, it is the same as --batch value')
    parser.add_argument('--model-class', type=str, default='vig', help='Model to be instantiated')
    return parser.parse_args()

def compute_split_sizes(split: str, size: int) -> List[int]:
    all_perc = tuple(int(x) if x.isdigit() else float(x) for x in split.split(','))
    if split.split(',')[0].isdigit():
        total = sum(all_perc)
        sizes = [x / total for x in all_perc]
    else:
        sizes = all_perc
    print(f'Sizes computed: {sizes}')
    return sizes

def normalize_image(data: Dict[str, torch.Tensor], mean: torch.Tensor, std: torch.Tensor) -> Dict[str, torch.Tensor]:
    if len(mean.shape) == 1:
        mean = mean.reshape(-1, 1, 1)
    if len(std.shape) == 1:
        std = std.reshape(-1, 1, 1)
        
    data['image'] = (data['image'] - mean) / std
    data['image'].clip(min=-5, max=5)
    return data

def parse_train_string(entry: str) -> List[str]:
    fields = entry.split('+')
    return fields

def export_model_path(folder: Path, path_to_save: str) -> None:
    count = 1
    output = {'best_model_path': path_to_save}
    while True:
        filename = f'checkpoint_savepath-v{count}.yaml' if count > 1 else 'checkpoint_savepath.yaml'
        p = folder / filename
        if p.is_file():
            count += 1
            continue
        with open(p, 'w') as fp:
            yaml.safe_dump(output, fp)
        break
    return

def load_checkpoint_from_file(f: Path) -> Path:
    with open(f, 'r') as fp:
        d = yaml.safe_load(fp)
    return Path(d['best_model_path'])

def get_encoder_state(d: Dict) -> Dict:
    return {x[8:]: d[x] for x in d if x.startswith('encoder')}

if __name__ == '__main__':
    args = parse()

    EPOCHS = args.n
    LR = args.lr
    BATCH = args.batch
    MODEL_CONFIG = Path(args.model_config) if not args.model_config is None else None
    MODEL_CLS = args.model_class
    COMET = args.comet
    DATASET = args.dataset
    SPLIT = args.split
    SEED = args.seed
    SPLIT_SEED = args.split_seed if not args.split_seed is None else args.seed
    PRETRAINED_CKPT = args.pretrained_checkpoint
    CKPT_FOLDER = args.checkpoint_folder
    EXP_NAME = args.exp_name
    DEBUG_RUN = args.debug
    TEST = args.test
    VAL_BATCH = args.val_batch
    
    RESUME = args.resume
    EXP_KEY = args.experiment_key
    
    if RESUME and EXP_KEY is None:
        raise ValueError(f'Invalid experiment key! {EXP_KEY}')

    if VAL_BATCH is None:
        VAL_BATCH = BATCH

    ckpt_path = Path(CKPT_FOLDER)
    pretrained_ckpt = Path(PRETRAINED_CKPT) if not PRETRAINED_CKPT is None else None

    if TEST is None and not ckpt_path.is_dir():
        ckpt_path.mkdir(parents=True)

    pl.seed_everything(SEED)

    basepath = Path.home() / 'datasets'
    dataset_folder = basepath / DATASET
    dataset_folder = str(dataset_folder)

    if MODEL_CONFIG is None:
        model_config = {
            'in_channels': 3,
            'out_channels': [128, 256, 512],
            'heads': 16,
            'n_classes': 1,
            'input_resolution': (256, 256),
            'reduce_factor': 2,
            'pyramid_reduction': 4,
            'act': 'relu',
            'k': 9,
            'overlapped_patch_emb': True,
        }
    else:
        with open(MODEL_CONFIG, 'r') as fp:
            model_config = yaml.safe_load(fp)
        model_config['lr'] = LR

    # create dataset
    other_params = {}
    if DATASET == 'patternnet':
        model_config['n_classes'] = 38
        model_config['metric_args'] = {
            'task': 'multiclass',
            'num_classes': 38,
            'average': 'micro'
        }
        mean_val = torch.tensor([91.6640, 91.9425, 81.3333])
        std_val = torch.tensor([49.9692, 47.3929, 45.5676])        
        
        dataset = PatternNet(root=dataset_folder, download=False, transforms=partial(normalize_image, mean=mean_val, std=std_val))
        assert not SPLIT is None
        sizes = compute_split_sizes(SPLIT, len(dataset))
        print(f'Split seed is set to {SPLIT_SEED}')
        train_dataset, val_dataset, test_dataset = random_split(dataset, sizes, generator=torch.Generator().manual_seed(SPLIT_SEED))
        del dataset
        other_params['drop_last'] = True
    elif DATASET == 'resisc45':
        model_config['n_classes'] = 45
        model_config['metric_args'] = {
            'task': 'multiclass',
            'num_classes': 45,
            'average': 'micro'
        }
        mean_val = torch.tensor([93.8935, 97.1123, 87.5696])
        std_val = torch.tensor([51.8668, 47.2381, 47.0614])
        
        train_dataset = RESISC45(root=dataset_folder, download=False, split='train', transforms=partial(normalize_image, mean=mean_val, std=std_val))
        val_dataset = RESISC45(root=dataset_folder, download=False, split='val', transforms=partial(normalize_image, mean=mean_val, std=std_val))
        test_dataset = RESISC45(root=dataset_folder, download=False, split='test', transforms=partial(normalize_image, mean=mean_val, std=std_val))
    elif DATASET == 'bigearthnet' or 'bigearthnet' in DATASET:
        model_config['n_classes'] = 43
        model_config['metric_args'] = {
            'task': 'multilabel',
            'num_labels': 43,
            'average': 'micro',
            'multidim_average': 'global'
        }

        mean_val = torch.tensor([352.7397,  441.2881,  624.5734,  601.4513,  960.7608, 1796.2574, 2076.4163, 2219.6677, 2265.7920, 2245.8081, 1585.1255, 1004.8402])
        std_val = torch.tensor([584.2036,  617.6658,  623.5845,  712.4464,  750.9664, 1096.1512, 1265.7113, 1373.5614, 1345.8695, 1289.3424, 1073.6807, 811.9662])

        if DATASET != 'bigearthnet':
            # this is a path
            dataset_folder = Path(DATASET)

        train_dataset = BigEarthNet(root=dataset_folder, download=False, bands='s2', split='train', num_classes=43, transforms=partial(normalize_image, mean=mean_val, std=std_val))
        val_dataset = BigEarthNet(root=dataset_folder, download=False, bands='s2', split='val', num_classes=43, transforms=partial(normalize_image, mean=mean_val, std=std_val))
        test_dataset = BigEarthNet(root=dataset_folder, download=False, bands='s2', split='test', num_classes=43, transforms=partial(normalize_image, mean=mean_val, std=std_val))

    print(f'Length of training dataset {len(train_dataset)}')
    print(f'Length of validation dataset {len(val_dataset)}')
    print(f'Length of test dataset {len(test_dataset)}')
    
    if MODEL_CLS == 'vig':
        model_config['enable_pos_encoding'] = True
        model = PyramidViGLT(**model_config)
    elif MODEL_CLS.startswith('resnet'):
        model = ResNetLT(resnet=MODEL_CLS, **model_config)
    elif MODEL_CLS == 'vit':
        model = ViTLT(**model_config)
    else:
        raise ValueError(f'Invalid model class {MODEL_CLS}')

    if not pretrained_ckpt is None:
        if pretrained_ckpt.suffix == '.yaml' or not str(pretrained_ckpt).endswith('ckpt'):
            if not str(pretrained_ckpt).endswith('yaml'):
                pretrained_ckpt = pretrained_ckpt / 'checkpoint_savepath.yaml'
            pretrained_ckpt = load_checkpoint_from_file(pretrained_ckpt)
        if not DEBUG_RUN:
            ckpt = torch.load(pretrained_ckpt)

            if MODEL_CLS == 'vig':
                incom_keys = model.model.encoder.load_state_dict(get_encoder_state(ckpt['state_dict']), strict=False)
            else:
                incom_keys = model.load_state_dict(ckpt['state_dict'], strict=False)
            print(f'Incompatible keys: {incom_keys}')
            del ckpt
        
    # create dataloaders
    train_loader = DataLoader(train_dataset, BATCH, shuffle=True, num_workers=8, **other_params)
    val_loader = DataLoader(val_dataset, VAL_BATCH, shuffle=False, num_workers=8, **other_params)
    test_loader = DataLoader(test_dataset, VAL_BATCH, shuffle=False, num_workers=8, **other_params)

    if COMET:
        logger_config = {
            'api_key': os.environ.get('COMET_API_KEY'),
            'project_name': 'ssl-vig',
        }
        if not RESUME:
            logger_config['experiment_name'] = f'{DATASET}-train-{EXP_NAME}'
        else:
            logger_config['experiment_key'] = EXP_KEY
    else:
        logger_config = None

    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, mode='min'),
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            dirpath=ckpt_path,
            save_last=True
        )
    ]

    trainer_args = {
        'accelerator': 'gpu',
        'devices': [0],
        'max_epochs': EPOCHS,
        'check_val_every_n_epoch': 1,
        'callbacks': callbacks,
        'fast_dev_run': DEBUG_RUN
    }
    
    if COMET:
        trainer_args['logger'] = CometLogger(**logger_config)

    trainer = pl.Trainer(**trainer_args)
    if TEST is None:
        fit_args = {}
        if RESUME:
            fit_args['ckpt_path'] = RESUME
            print(f'Setting ckpt path to {RESUME}')
        trainer.fit(model, train_loader, val_loader, **fit_args)

        trainer_args['logger'].experiment.log_model(
            'best model', trainer.checkpoint_callback.best_model_path
        )

        best_model_path = callbacks[1].best_model_path
        export_model_path(ckpt_path, best_model_path)

    test_args = {}
    if not TEST is None:
        if MODEL_CLS == 'vig':
            model = PyramidViGLT(**model_config)
        elif MODEL_CLS.startswith('resnet'):
            model = ResNetLT(MODEL_CLS, **model_config)
        elif MODEL_CLS == 'vit':
            model = ViTLT(**model_config)
        state = torch.load(TEST)
        model.load_state_dict(state['state_dict'])
    else:
        test_args['ckpt_path'] = 'best'

    model.eval()
    trainer.test(model, test_loader, **test_args)

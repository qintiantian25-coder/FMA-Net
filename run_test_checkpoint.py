import os
import argparse
import torch

from config import Config
from model import FMANet
from train import Trainer
from data_blindpixel import get_dataset


def load_checkpoint_to_trainer(trainer, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Common project save format: {'model_D_state_dict': ..., 'model_R_state_dict': ..., ...}
    if isinstance(ckpt, dict):
        if 'model_D_state_dict' in ckpt:
            try:
                trainer.model.degradation_learning_network.load_state_dict(ckpt['model_D_state_dict'])
            except Exception as e:
                print(f'[!] Warning: failed to load model_D_state_dict: {e}')
        if 'model_R_state_dict' in ckpt and hasattr(trainer.model, 'restoration_network'):
            try:
                trainer.model.restoration_network.load_state_dict(ckpt['model_R_state_dict'])
            except Exception as e:
                print(f'[!] Warning: failed to load model_R_state_dict: {e}')

        # Some checkpoints store a single 'model' key containing a full state dict
        if 'model' in ckpt and isinstance(ckpt['model'], dict):
            state = ckpt['model']
            # Try to load into whole FMANet (non-strict to allow partial matches)
            try:
                trainer.model.load_state_dict(state, strict=False)
            except Exception:
                pass

    else:
        # If checkpoint is a raw state_dict, try to load into restoration and degradation nets
        try:
            trainer.model.degradation_learning_network.load_state_dict(ckpt)
        except Exception:
            try:
                trainer.model.load_state_dict(ckpt, strict=False)
            except Exception as e:
                print(f'[!] Warning: failed to load checkpoint into model: {e}')

    print(f'[*] Loaded checkpoint from {ckpt_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./experiment.cfg')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint .pt')
    parser.add_argument('--out_dir', type=str, default=None, help='optional override of config.save_dir for outputs')
    args = parser.parse_args()

    config = Config(args.config_path)
    if args.out_dir:
        config.save_dir = args.out_dir

    # build model and trainer (Trainer will .cuda() the model)
    model = FMANet(config=config)
    trainer = Trainer(config=config, model=model)

    # load checkpoint weights into trainer.model
    load_checkpoint_to_trainer(trainer, args.checkpoint)

    # Run test dataloader and inference using Trainer.test (test logic lives in train.py)
    test_loader = get_dataset(config, type='test')
    print('===> Running Trainer.test (inference + optional selective feathering) ...')
    trainer.test(test_loader)

    # Run quantitative evaluation (saves CSV under save_dir/blind_eval)
    gt_root = os.path.join(config.dataset_path, 'test_sharp')
    output_root = os.path.join(config.save_dir, 'test')
    print('===> Running Trainer.test_quantitative_result ...')
    trainer.test_quantitative_result(gt_dir=gt_root, output_dir=output_root, image_border=0)
    print('===> Done.')


if __name__ == '__main__':
    main()


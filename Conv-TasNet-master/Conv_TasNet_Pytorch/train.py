import torch
import argparse
import sys
sys.path.append('C:\\Users\\Randy\\Downloads\\Conv-TasNet-master\\Conv-TasNet-master\\Conv_TasNet_Pytorch\\options')
from trainer import Trainer
from Conv_TasNet import ConvTasNet
from DataLoaders import make_dataloader
from options.option import parse
from utils import get_logger

def main():
    # Reading options
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to options YAML file.')
    args = parser.parse_args()

    opt = parse(args.opt, is_tain=True)
    logger = get_logger(__name__)
    
    logger.info('Building the model of Conv-TasNet')
    net = ConvTasNet(**opt['net_conf'])

    logger.info('Building the trainer of Conv-TasNet')
    gpuid = tuple(opt['gpu_ids'])
    trainer = Trainer(net, **opt['train'], resume=opt['resume'],
                      gpuid=gpuid, optimizer_kwargs=opt['optimizer_kwargs'])

    logger.info('Making the train and test data loader')
    train_loader = make_dataloader(is_train=True, data_kwargs=opt['datasets']['train'], num_workers=opt['datasets']
                                   ['num_workers'], chunk_size=opt['datasets']['chunk_size'], batch_size=opt['datasets']['batch_size'])
    val_loader = make_dataloader(is_train=False, data_kwargs=opt['datasets']['val'], num_workers=opt['datasets']
                                   ['num_workers'], chunk_size=opt['datasets']['chunk_size'], batch_size=opt['datasets']['batch_size'])
    logger.info('Train data loader: {}, Test data loader: {}'.format(len(train_loader), len(val_loader)))
    trainer.run(train_loader,val_loader)


if __name__ == "__main__":
    main()

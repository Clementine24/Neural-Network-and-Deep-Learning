import os
import json
import math
import torch
import random
import sys
import logging
import time
import numpy as np

from data import build_loaders
from util.config import Config
from torch.utils.tensorboard import SummaryWriter
from models import get_trainer

torch.backends.cudnn.benchnark=True
# python train inpaint.yml
args = Config(sys.argv[2])
logger = logging.getLogger(__name__)
time_stamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_dir = 'model_logs/{}/{}_{}'.format(args.trainer, time_stamp,  args.log_dir)
args.log_id = os.path.join(time_stamp,  args.log_dir)
result_dir = 'result_logs/{}/{}_{}'.format(args.trainer, time_stamp, args.log_dir)
tensorboard_logger = SummaryWriter(log_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


torch.backends.cudnn.benchmark = True

def logger_init():
    """
    Initialize the logger to some file.
    """
    logpath = 'logs/{}'.format(args.trainer)
    if not os.path.exists(logpath):
        os.makedirs(logpath)

    logging.basicConfig(level=logging.INFO)

    logfile = 'logs/{}/{}_{}.log'.format(args.trainer, time_stamp, args.log_dir)
    fh = logging.FileHandler(logfile, mode='w')
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def train():
    logger_init()
    dataset_type = args.dataset

    batch_size = args.batch_size
    logger.info(str(args))

    # Dataset setting
    logger.info("Initialize the dataset...")
    vocab, train_loader, train_check_loader, val_loader, val_check_loader, iter_per_epoch = build_loaders(args, logger)
    datas = {
         'vocab':vocab,
         'train_loader':train_loader,
         'val_loader':val_loader,
         'train_check_loader':train_check_loader,
         'val_check_loader':val_check_loader,
         'iter_per_epoch':iter_per_epoch
        }
    logger.info("Finish the dataset initialization.")

    # Construct a Trainer
    trainer = get_trainer(args.trainer, args, datas, log_dir, logger, tensorboard_logger)
    trainer.train()

def test():
    logger_init()
    dataset_type = args.dataset

    batch_size = args.batch_size
    logger.info(str(args))

    # Dataset setting
    logger.info("Initialize the dataset...")
    vocab, train_loader, train_check_loader, val_loader, val_check_loader, iter_per_epoch = build_loaders(args, logger)
    #print(len(train_loader), len(val_loader))
    datas = {
         'vocab':vocab,
         'train_loader':train_loader,
         'val_loader':val_loader,
         'train_check_loader':train_check_loader,
         'val_check_loader':val_check_loader,
         'iter_per_epoch':iter_per_epoch
        }
    logger.info("Finish the dataset initialization.")

    # Construct a Trainer
    trainer = get_trainer(args.trainer, args, datas, log_dir, logger, tensorboard_logger)
    trainer.validate()
    
    
def generate():
    logger_init()
    logger.info(str(args))
    
    mask_probs = np.linspace(0.05, 0.15, 10)
    
    # Dataset setting
    logger.info("Initialize the dataset...")
    vocab, train_loader, train_check_loader, val_loader, val_check_loader, iter_per_epoch = build_loaders(args, logger, shuffle=False)
    # print(len(train_loader), len(val_loader))
    datas = {
        'vocab': vocab,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'train_check_loader': train_check_loader,
        'val_check_loader': val_check_loader,
        'iter_per_epoch': iter_per_epoch
    }
    logger.info("Finish the dataset initialization.")
    
    # Construct a Trainer
    trainer = get_trainer(args.trainer, args, datas, log_dir, logger, tensorboard_logger)
    # trainer.generate(mask_prob=0.15)

    for mask_prob in mask_probs:
        logger.info("Initialize the dataset...")
        vocab, train_loader, train_check_loader, val_loader, val_check_loader, iter_per_epoch = build_loaders(args, logger, shuffle=False, mask_prob=1-mask_prob)
        datas = {
            'vocab': vocab,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'train_check_loader': train_check_loader,
            'val_check_loader': val_check_loader,
            'iter_per_epoch': iter_per_epoch
        }
        logger.info("Finish the dataset initialization.")
        trainer.datas = datas
        trainer.generate(mask_prob=mask_prob)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'train':
        train()
    elif mode == 'test':
        test()
    elif mode == "generate":
        generate()

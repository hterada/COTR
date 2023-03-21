import os
import math
import abc
import time
from typing import Optional

import tqdm
import torch.nn as nn
import tensorboardX

from COTR.models import COTR
from COTR.trainers import tensorboard_helper
from COTR.utils import utils
from COTR.utils.utils import TR
from COTR.utils.line_profiler_header import *
from COTR.options import options_utils


class BaseDistiller(abc.ABC):
    '''base distiller class.
    contains methods for training, validation, and writing output.
    '''

    def __init__(self, t_model:COTR, s_model:COTR,
                optimizer, criterion,
                train_loader, val_loader,
                use_cuda:bool, out_dir:str, tb_out_dir:str, max_iter:int, valid_iter:int,
                opt_str:str,
                resume:bool,
                t_weights_path:str,
                s_weights_path:Optional[str]=None):
        self.use_cuda = use_cuda
        self.t_model = t_model
        self.s_model = s_model
        # backbone
        self.t_backbone = self.get_t_backbone()
        self.s_backbone = self.get_s_backbone()

        self.optim = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out = out_dir
        if not os.path.exists(self.out):
            os.makedirs(self.out)
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.valid_iter = valid_iter
        self.tb_pusher = tensorboard_helper.TensorboardPusher.create(tb_out_dir)
        self.push_opt_to_tb(opt_str)
        # resume
        self.need_resume = resume
        if self.need_resume:
            assert s_weights_path is not None
            self.resume( s_weights_path )

        self.t_weights_path = t_weights_path
        self.s_weights_path = s_weights_path

        # self.load_pretrained_weights(self.t_weights_path, self.s_weights_path)

    def push_opt_to_tb(self, opt_str:str):
        # opt_str = options_utils.opt_to_string(self.opt)
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(False)
        tb_datapack.set_iteration(self.iteration)
        tb_datapack.add_text({'options': opt_str})
        self.tb_pusher.push_to_tensorboard(tb_datapack)

    @abc.abstractmethod
    def get_t_backbone(self)->Optional[nn.Module]:
        """Get teacher's backbone

        Returns:
            Optional[nn.Module]: _description_
        """
        return None

    @abc.abstractmethod
    def get_s_backbone(self)->Optional[nn.Module]:
        """Get student's backbone

        Returns:
            Optional[nn.Module]: _description_
        """
        return None

    @abc.abstractmethod
    def load_pretrained_weights(self, t_weights_path:str, s_weights_path:Optional[str]):
        pass

    @abc.abstractmethod
    def validate_batch(self, data_pack):
        pass

    @abc.abstractmethod
    def validate(self):
        pass

    @abc.abstractmethod
    def train_batch(self, data_pack):
        '''train for one batch of data
        '''
        pass

    def train_epoch(self):
        '''train for one epoch
        one epoch is iterating the whole training dataset once
        '''
        TR("train_epoch")
        # training mode
        self.s_backbone.train()
        print(f"trail_loader len:{len(self.train_loader)}")
        for batch_idx, data_pack in tqdm.tqdm(enumerate(self.train_loader),
                                              initial=self.iteration % len(
                                                  self.train_loader),
                                              total=len(self.train_loader),
                                              desc='Train epoch={0}'.format(
                                                  self.epoch),
                                              ncols=80,
                                              leave=True,
                                              ):

            # iteration = batch_idx + self.epoch * len(self.train_loader)
            # if self.iteration != 0 and (iteration - 1) != self.iteration:
            #     continue  # for resuming
            # self.iteration = iteration
            # self.iteration += 1
            if self.iteration % self.valid_iter == 0:
                time.sleep(2)  # Prevent possible deadlock during epoch transition TODO: ?
                self.validate()
            self.train_batch(data_pack)

            if self.iteration >= self.max_iter:
                break
            self.iteration += 1

    def train(self):
        '''entrance of the whole training process
        '''
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch,
                                 max_epoch,
                                 desc='Train',
                                 ncols=80):
            self.epoch = epoch
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break

    @abc.abstractmethod
    def resume(self, s_weights_path:str):
        pass

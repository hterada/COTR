from typing import Optional
import os
import math
import os.path as osp
import time

import tqdm
import torch
import torch.nn as nn
import numpy as np
import torchvision.utils as vutils
from torchinfo import summary
from PIL import Image, ImageDraw


from COTR.models import COTR
from COTR.utils import utils, debug_utils, constants
from COTR.utils.utils import TR,TR1
from COTR.utils.line_profiler_header import *
from COTR.trainers import tensorboard_helper, base_distiller
from COTR.projector import pcd_projector
from COTR.models.misc import NestedTensor

class COTRDistiller(base_distiller.BaseDistiller):
    def __init__(self, t_model:COTR, s_model:COTR,
                optimizer, criterion,
                train_loader, val_loader,
                use_cuda:bool, device:torch.device, out_dir:str, tb_out_dir:str, max_iter:int, valid_iter:int,
                opt_str:str,
                resume:bool,
                t_weights_path:str, s_weights_path:Optional[str]=None):

        super().__init__(t_model, s_model, optimizer, criterion,
                         train_loader, val_loader, use_cuda, device, out_dir, tb_out_dir,
                         max_iter, valid_iter,
                         opt_str,
                         resume, t_weights_path, s_weights_path)
        self.summary_done = False

    def get_t_backbone(self)->Optional[nn.Module]:
        return self.t_model.backbone[0]

    def get_s_backbone(self)->Optional[nn.Module]:
        return self.s_model.backbone[0]

    @staticmethod
    def cycle_consistency_bidirectional(model, img, query, pred)->float:
        cycle = model(img, pred)['pred_corrs']
        return torch.nn.functional.mse_loss(cycle, query).item()

    @profile
    def validate_batch(self, data_pack):
        assert self.t_backbone.training is False
        assert self.s_backbone.training is False

        with torch.no_grad():
            img = data_pack['image'].cuda(self.device)
            b,c,h,w = img.shape
            query = data_pack['queries'].cuda(self.device)
            # target = data_pack['targets'].cuda(self.device)
            s_pred = self.s_model(img, query)['pred_corrs']
            t_pred = self.t_model(img, query)['pred_corrs']

            self.optim.zero_grad()
            mask = torch.ones((b,h,w), dtype=torch.bool, device=img.device)
            nested_tensor = NestedTensor(img, mask)
            # student
            sb_pred = self.s_backbone(nested_tensor)
            # teacher
            tb_pred = self.t_backbone(nested_tensor)

            loss = torch.nn.functional.mse_loss(sb_pred['0'].tensors, tb_pred['0'].tensors)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                print('loss is nan while validating')

            # cycle consistency
            t_cc = self.__class__.cycle_consistency_bidirectional(self.t_model, img, query, t_pred)
            s_cc = self.__class__.cycle_consistency_bidirectional(self.s_model, img, query, s_pred)

            return loss_data, s_pred, t_cc, s_cc

    def validate(self):
        '''validate for whole validation dataset
        '''
        # inferring mode
        self.t_backbone.eval()
        self.s_backbone.eval()

        val_loss_list = []
        t_cc_list = []
        s_cc_list = []
        for batch_idx, data_pack in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            # validate a batch
            loss_data, pred, t_cc, s_cc = self.validate_batch(data_pack)
            val_loss_list.append(loss_data)
            t_cc_list.append(t_cc)
            s_cc_list.append(s_cc)

        mean_loss = np.array(val_loss_list).mean()
        mean_t_cc = np.array(t_cc_list).mean()
        mean_s_cc = np.array(s_cc_list).mean()
        validation_data = {'val_loss': mean_loss, 'pred': pred, 't_cc':mean_t_cc, 's_cc':mean_s_cc}
        self.push_validation_data(data_pack, validation_data)
        self.save_model()

        # training mode
        self.s_backbone.train()

    def push_validation_data(self, data_pack, validation_data):
        val_loss = validation_data['val_loss']
        t_cc = validation_data['t_cc']
        s_cc = validation_data['s_cc']
        pred_corrs = np.concatenate([data_pack['queries'].numpy(), validation_data['pred'].cpu().numpy()], axis=-1)
        pred_corrs = self.draw_corrs(data_pack['image'], pred_corrs)
        gt_corrs = np.concatenate([data_pack['queries'].numpy(), data_pack['targets'].cpu().numpy()], axis=-1)
        gt_corrs = self.draw_corrs(data_pack['image'], gt_corrs, (0, 255, 0))

        gt_img = vutils.make_grid(gt_corrs, normalize=True, scale_each=True)
        pred_img = vutils.make_grid(pred_corrs, normalize=True, scale_each=True)
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(False)
        tb_datapack.set_iteration(self.iteration)
        tb_datapack.add_scalar({'loss/val': val_loss})
        tb_datapack.add_scalar({'cc': {'val/teacher':t_cc, 'val/student':s_cc}})
        tb_datapack.add_image({'image/gt_corrs': gt_img})
        tb_datapack.add_image({'image/pred_corrs': pred_img})
        self.tb_pusher.push_to_tensorboard(tb_datapack)

    def save_model(self):
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.s_model.state_dict(),
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if self.iteration % (10 * self.valid_iter) == 0:
            torch.save({
                'epoch': self.epoch,
                'iteration': self.iteration,
                'optim_state_dict': self.optim.state_dict(),
                'model_state_dict': self.s_model.state_dict(),
            }, osp.join(self.out, f'{self.iteration}_checkpoint.pth.tar'))

    def draw_corrs(self, imgs, corrs, col=(255, 0, 0)):
        imgs = utils.torch_img_to_np_img(imgs)
        out = []
        for img, corr in zip(imgs, corrs):
            img = np.interp(img, [img.min(), img.max()], [0, 255]).astype(np.uint8)
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            corr *= np.array([constants.MAX_SIZE * 2, constants.MAX_SIZE, constants.MAX_SIZE * 2, constants.MAX_SIZE])
            for c in corr:
                draw.line(c, fill=col)
            out.append(np.array(img))
        out = np.array(out) / 255.0
        return utils.np_img_to_torch_img(out)


    def train_batch(self, data_pack):
        '''train for one batch of data
        '''
        TR()
        assert self.t_backbone.training == False
        img = data_pack['image'].cuda(self.device)
        b,c,h,w = img.shape
        # img.shape = (24, 3, 256, 512)
        TR(f"img:{type(img), img.shape}")
        # query = data_pack['queries'].cuda(self.device)
        # target = data_pack['targets'].cuda(self.device)

        self.optim.zero_grad()
        mask = torch.ones((b,h,w), dtype=torch.bool, device=img.device)
        nested_tensor = NestedTensor(img, mask)
        # student
        sb_pred = self.s_backbone(nested_tensor)
        assert list(sb_pred.keys())==['0']

        # teacher
        tb_pred = self.t_backbone(nested_tensor)
        assert list(tb_pred.keys())==['0']

        loss = torch.nn.functional.mse_loss(sb_pred['0'].tensors, tb_pred['0'].tensors)

        # if self.opt.cycle_consis and self.opt.bidirectional:
        #     cycle = self.model(img, pred)['pred_corrs']
        #     mask = torch.norm(cycle - query, dim=-1) < 10 / constants.MAX_SIZE
        #     if mask.sum() > 0:
        #         cycle_loss = torch.nn.functional.mse_loss(cycle[mask], query[mask])
        #         loss += cycle_loss
        # elif self.opt.cycle_consis and not self.opt.bidirectional:
        #         img_reverse = torch.cat([img[..., constants.MAX_SIZE:], img[..., :constants.MAX_SIZE]], axis=-1)
        #         query_reverse = pred.clone()
        #         query_reverse[..., 0] = query_reverse[..., 0] - 0.5
        #         cycle = self.model(img_reverse, query_reverse)['pred_corrs']
        #         cycle[..., 0] = cycle[..., 0] - 0.5
        #         mask = torch.norm(cycle - query, dim=-1) < 10 / constants.MAX_SIZE
        #         if mask.sum() > 0:
        #             cycle_loss = torch.nn.functional.mse_loss(cycle[mask], query[mask])
        #             loss += cycle_loss
        loss_data = loss.data.item()
        if np.isnan(loss_data):
            print('loss is nan during training')
            self.optim.zero_grad()
        else:
            loss.backward()

            # log
            ## cycle consistency
            with torch.no_grad():
                query = data_pack['queries'].cuda(self.device)
                TR(f"query:{type(query), query.shape}")
                s_pred = self.s_model(img, query)['pred_corrs']
                t_pred = self.t_model(img, query)['pred_corrs']

                t_cc = self.__class__.cycle_consistency_bidirectional(self.t_model, img, query, t_pred)
                s_cc = self.__class__.cycle_consistency_bidirectional(self.s_model, img, query, s_pred)

                # summary
                if not self.summary_done:
                    TR1()
                    self.summary_done = True
                    TR("summary t_model")
                    summary(self.t_model, input_data=[img[:1,...], query[:1,...]], device=self.device)
                    TR("summary s_model")
                    summary(self.s_model, input_data=[img[:1,...], query[:1,...]], device=self.device)

                # ccd = np.linalg.norm(t_cc-s_cc)

            self.push_training_data(tb_pred['0'].tensors, sb_pred['0'].tensors, loss, t_cc, s_cc )
        self.optim.step()

    def push_training_data(self, t_pred, s_pred, loss, t_cc, s_cc):
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(True)
        tb_datapack.set_iteration(self.iteration)
        tb_datapack.add_histogram({'distribution/s_pred': s_pred})
        tb_datapack.add_histogram({'distribution/t_pred': t_pred})
        tb_datapack.add_scalar({'loss/train': loss})
        tb_datapack.add_scalar({'cc': {'train/teacher':t_cc, 'train/student':s_cc} })
        self.tb_pusher.push_to_tensorboard(tb_datapack)

    def resume(self, s_weights_path:str): #TODO:
        '''resume training:
        resume from the recorded epoch, iteration, and saved weights.
        resume from the model with the same name.

        Arguments:
            opt {[type]} -- [description]
        '''
        TR1('resume distill')
        # 1. load check point
        if os.path.isfile(s_weights_path):
            checkpoint = torch.load(s_weights_path)
        else:
            raise FileNotFoundError(
                f'model check point cannnot found: {s_weights_path}')
        # 2. load data
        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        # self.load_pretrained_weights()
        self.optim.load_state_dict(checkpoint['optim_state_dict'])

    def load_pretrained_weights(self, t_weights_path:str, s_weights_path:Optional[str]):
        '''
        load pretrained weights from another model
        '''
        assert t_weights_path is not None
        assert os.path.isfile(t_weights_path), f"NOT a file:{t_weights_path}"

        # teacher
        def load( weights_path, model, sym:str):
            print(f"Load pretrained ({sym}) weights from {weights_path}...")
            saved_weights = torch.load(weights_path)['model_state_dict']
            if utils.safe_load_weights(model, saved_weights)==True:
                content_list = []
                content_list += [f'Loaded pretrained ({sym}) weights from {weights_path}']
                utils.print_notification(content_list)

        load( t_weights_path, self.t_model, 't' )

        # student
        if s_weights_path is not None:
            assert os.path.isfile(s_weights_path), f"NOT a file:{s_weights_path}"
            load( s_weights_path, self.s_model, 's' )


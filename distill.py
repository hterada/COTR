print("*** distill ***")
import argparse
import subprocess
import pprint
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from COTR.models import build_model, COTR
from COTR.utils import debug_utils, utils, constants
from COTR.utils.utils import TR, TR1
from COTR.utils.line_profiler_header import *

from COTR.datasets import cotr_dataset
from COTR.trainers.cotr_distiller import COTRDistiller
from COTR.global_configs import general_config
from COTR.options.options import *
from COTR.options.options_utils import *

from torchvision.models._utils import IntermediateLayerGetter

utils.fix_randomness(0)

def print_mem_capa(model:nn.Module, name:str):
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    print(f"{name}.mem_params = {mem_params/1024}[KB]")
    print(f"{name}.mem_bufs = {mem_bufs/1024}[KB]")


@profile
def distill_backbone(opt, gpuno:int, t_weights_path:str, s_weights_path:str, s_layer:str):
    # print env info
    pprint.pprint(dict(os.environ), width=1)
    result = subprocess.Popen(["nvidia-smi"], stdout=subprocess.PIPE)
    assert result.stdout is not None
    print(result.stdout.read().decode())
    device = torch.device(f"cuda:{gpuno}")
    print(f'can see {torch.cuda.device_count()} gpus')
    print(f'using gpu at "{device}" -- "{torch.cuda.get_device_name(device)}"')


    torch.cuda.empty_cache()

    ##########################
    # setup teacher model
    TR1()
    t_model:COTR = build_model(opt)
    t_model = t_model.cuda(device)

    ## load teacher weights
    weights = torch.load(t_weights_path, map_location='cpu')['model_state_dict']
    utils.safe_load_weights(t_model, weights)

    # eval(): switch to inference mode
    t_model = t_model.eval()

    ## 教師モデルは勾配計算させない。
    for param in t_model.parameters():
        param.requires_grad = False

    ##########################
    # setup student model
    s_model:COTR = build_model(opt)
    s_model = s_model.cuda(device)

    ## modify body layer
    body = IntermediateLayerGetter(s_model.backbone[0].body, {s_layer: "0"}) #new_name = "0"
    # print(f"s_body0:{body}")
    print(f"s_body0.return_layers:{body.return_layers}")

    ## check body's output shape
    REF = constants.MAX_SIZE
    dm_in = torch.randn(1, 3, REF, REF).cuda(device)
    dm_ot = body(dm_in)
    assert len(dm_ot)==1
    TR(f"dm_ot:{dm_ot['0'].shape}" )
    n,c,h,w = dm_ot['0'].shape

    ## Convolution により出力サイズを (N, 1024, 16, 16)にする
    TGT = 16
    assert h >= TGT
    assert (h%TGT)==0
    assert h/TGT == w/TGT
    conv_size = int(h/TGT)
    conv = torch.nn.Conv2d(c, 1024, (conv_size,conv_size), stride=conv_size)
    TR(f"student additional conv:{conv}")
    seq_conv = torch.nn.Sequential(conv)
    # MEMO:>>
    body['resize'] = seq_conv
    body.return_layers = {"resize":"0"}
    s_model.backbone[0].body = body
    # MEMO:<<
    s_model = s_model.cuda(device)

    ## load student weights
    weights = torch.load(s_weights_path, map_location='cpu')['model_state_dict']
    utils.safe_load_weights(s_model, weights)

    # backbone 以外の学習を止める
    for module in [s_model.transformer, s_model.corr_embed, s_model.query_proj, s_model.input_proj]:
        for param in module.parameters():
            param.requires_grad = False
    # backbone は学習させる
    for param in s_model.backbone.parameters():
        param.requires_grad = True

    #
    # for name, param in s_model.named_parameters():
    #     print(f"s_model.{name} {param.requires_grad}")
    print_mem_capa(t_model, "t_model")
    print_mem_capa(s_model, "s_model")

    # print(f"s_model.backbone:{s_model.backbone[0]}")
    s_model.cuda(device)
    s_model.train(True)

    #
    if opt.enable_zoom:
        train_dset = cotr_dataset.COTRZoomDataset(opt, 'train')
        val_dset = cotr_dataset.COTRZoomDataset(opt, 'val')
    else:
        TR("train dset")
        train_dset = cotr_dataset.COTRDataset(opt, 'train')
        TR("val dset")
        val_dset = cotr_dataset.COTRDataset(opt, 'val')

    print(f"val_dset:{len(val_dset)}")

    train_loader = DataLoader(train_dset, batch_size=opt.batch_size,
                              shuffle=opt.shuffle_data, num_workers=opt.workers,
                              worker_init_fn=utils.worker_init_fn, pin_memory=True)
    val_loader = DataLoader(val_dset, batch_size=opt.batch_size,
                            shuffle=opt.shuffle_data, num_workers=opt.workers,
                            drop_last=True, worker_init_fn=utils.worker_init_fn, pin_memory=True)
    # optimizer
    optim_list = [
        {"params": s_model.backbone.parameters(), "lr": opt.learning_rate},
    ]
    optim = torch.optim.Adam(optim_list)

    opt_str = options_utils.opt_to_string(opt)

    # distiller
    distiller = COTRDistiller(t_model, s_model,
                              optim, None, train_loader, val_loader,
                              opt.use_cuda, device, opt.out, opt.tb_out, opt.max_iter, opt.valid_iter,
                              opt_str,
                              opt.resume, t_weights_path, s_weights_path
                              )
    distiller.train(opt.suffix)


if __name__ == "__main__":
    TR1()
    parser = argparse.ArgumentParser()

    set_general_arguments(parser)
    set_dataset_arguments(parser)
    set_nn_arguments(parser)
    set_COTR_arguments(parser)
    parser.add_argument('--num_kp', type=int,
                        default=100)
    parser.add_argument('--kp_pool', type=int,
                        default=100)
    parser.add_argument('--enable_zoom', type=str2bool,
                        default=False)
    parser.add_argument('--zoom_start', type=float,
                        default=1.0)
    parser.add_argument('--zoom_end', type=float,
                        default=0.1)
    parser.add_argument('--zoom_levels', type=int,
                        default=10)
    parser.add_argument('--zoom_jitter', type=float,
                        default=0.5)

    parser.add_argument('--out_dir', type=str,
                        default=general_config['d_out'], help='out directory')
    parser.add_argument(
        '--tb_dir', type=str, default=general_config['d_tb_out'], help='tensorboard runs directory')

    parser.add_argument('--learning_rate', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch size for training')
    parser.add_argument('--cycle_consis', type=str2bool, default=True,
                        help='cycle consistency')
    parser.add_argument('--bidirectional', type=str2bool, default=True,
                        help='left2right and right2left')
    parser.add_argument('--max_iter', type=int,
                        default=200000, help='total training iterations')
    parser.add_argument('--valid_iter', type=int,
                        default=1000, help='iterval of validation')
    parser.add_argument('--resume', type=str2bool, default=False,
                        help='resume distilling with `load_s_weights` model name')
    parser.add_argument('--cc_resume', type=str2bool, default=False,
                        help='resume from last run if possible')
    parser.add_argument('--need_rotation', type=str2bool, default=False,
                        help='rotation augmentation')
    parser.add_argument('--max_rotation', type=float, default=0,
                        help='max rotation for data augmentation')
    parser.add_argument('--rotation_chance', type=float, default=0,
                        help='the probability of being rotated')

    parser.add_argument('--load_t_weights', type=str, default=None, required=True,
                        help='load a pretrained set of weights for teacher, you need to provide the model id')
    parser.add_argument('--load_s_weights', type=str, default=None,
                        help='load a pretrained set of weights for student, you need to provide the model id')

    parser.add_argument('--suffix', type=str, default=None, help='model suffix')
    parser.add_argument('--gpuno', type=int, default=0, help='specify gpu number:0,1,...')

    opt = parser.parse_args()
    opt.command = ' '.join(sys.argv)

    layer_2_channels = {'layer1': 256,
                        'layer2': 512,
                        'layer3': 1024,
                        'layer4': 2048, }
    opt.dim_feedforward = layer_2_channels[opt.layer]
    opt.num_queries = opt.num_kp

    opt.name = get_compact_naming_cotr_distilled(opt)
    opt.out = os.path.join(opt.out_dir, opt.name)
    opt.tb_out = os.path.join(opt.tb_dir, opt.name)

    s_weights_path = None
    t_weights_path = None

    if opt.resume: # resume のほうが cc_resume より優先
        print('resuming from specified model')
        s_weights_path = os.path.join(opt.load_s_weights, 'checkpoint.pth.tar')
    else:
        if opt.cc_resume:
            # さっきの続きから。
            if os.path.isfile(os.path.join(opt.out, 'checkpoint.pth.tar')):
                print('resuming from last run')
                s_weights_path = os.path.join(opt.out, 'checkpoint.pth.tar')
                opt.load_s_weights = None
                opt.resume = True
            else:
                opt.resume = False
    print(f"s_weights_path={s_weights_path}")

    # teacher
    if opt.load_t_weights:
        t_weights_path = os.path.join(opt.load_t_weights, 'checkpoint.pth.tar')

    # student
    if opt.load_s_weights:
        s_weights_path = os.path.join(opt.load_s_weights, 'checkpoint.pth.tar')

    TR1()
    opt.scenes_name_list = build_scenes_name_list_from_opt(opt)
    TR1()

    if opt.confirm:
        confirm_opt(opt)
    else:
        print_opt(opt)

    save_opt(opt)
    TR1()
    distill_backbone(opt, opt.gpuno, t_weights_path, s_weights_path, opt.s_layer)

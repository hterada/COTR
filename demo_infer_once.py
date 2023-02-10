'''
COTR demo for human face
We use an off-the-shelf face landmarks detector: https://github.com/1adrianb/face-alignment
'''
import argparse
import os
import time

import cv2
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
# import torchprof

from COTR.utils import utils, debug_utils
from COTR.utils.stopwatch import StopWatch

from COTR.models import build_model
from COTR.options.options import *
from COTR.options.options_utils import *
from COTR.inference.inference_helper import triangulate_corr
from COTR.inference.sparse_engine import SparseEngine

from pytorch_memlab import MemReporter

utils.fix_randomness(0)
torch.set_grad_enabled(False)


def main(opt):
    model = build_model(opt)
    model = model.cuda()
    weights = torch.load(opt.load_weights_path, map_location='cpu')['model_state_dict']
    utils.safe_load_weights(model, weights)

    # eval(): switch to inference mode
    model = model.eval()
    # mem_rep = MemReporter(model)
    # # print(">>>>>> MemReporter 1.")
    # mem_rep.report()

    img_a = imageio.imread('./sample_data/imgs/face_1.png', pilmode='RGB')
    img_b = imageio.imread('./sample_data/imgs/face_2.png', pilmode='RGB')
    queries = np.load('./sample_data/face_landmarks.npy')[0]
    print(f"queries:len={len(queries)}\n{queries}")

    engine = SparseEngine(model, 32, mode='stretching')
    # with torchprof.Profile(model, use_cuda=True, profile_memory=True) as prof:
    with StopWatch("cotr_corr_once") as sw:
        # corrs: ndarray
        corrs = engine.cotr_corr_once(img_a, img_b, queries=queries)
    df = sw.to_DataFrame()
    df.to_csv("out/demo_infer_once_sw.csv", sep=",")
    print(df)

    # print(">>>>>> MemReporter 2.")
    # mem_rep.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_COTR_arguments(parser)
    parser.add_argument('--out_dir', type=str, default=general_config['out'], help='out directory')
    parser.add_argument('--load_weights', type=str, default=None, help='load a pretrained set of weights, you need to provide the model id')

    opt = parser.parse_args()
    opt.command = ' '.join(sys.argv)

    layer_2_channels = {'layer1': 256,
                        'layer2': 512,
                        'layer3': 1024,
                        'layer4': 2048, }
    opt.dim_feedforward = layer_2_channels[opt.layer]
    if opt.load_weights:
        opt.load_weights_path = os.path.join(opt.out_dir, opt.load_weights, 'checkpoint.pth.tar')
    print_opt(opt)
    main(opt)

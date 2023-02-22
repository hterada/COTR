#!/bin/bash

#
python distill.py --load_weights="default" --scene_file sample_data/jsons/debug_megadepth.json --info_level=rgbd --use_ram=yes --use_cc=no --batch_size=24 --learning_rate=1e-4 --lr_backbone=0 --max_iter=300000 --workers=8 --cycle_consis=yes --bidirectional=yes --position_embedding=lin_sine --layer=layer3 --s_layer=layer2 --confirm=no --dataset_name=megadepth_sushi --suffix=distill_1 --valid_iter=1000 --enable_zoom=no --crop_cam=crop_center_and_resize --out_dir=./out --resume=no


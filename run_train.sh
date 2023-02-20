#!/bin/bash

# org
#python train_cotr.py --scene_file sample_data/jsons/200_megadepth.json --info_level=rgbd --use_ram=yes --use_cc=no --batch_size=24 --learning_rate=1e-4 --lr_backbone=0 --max_iter=300000 --workers=8 --cycle_consis=yes --bidirectional=yes --position_embedding=lin_sine --layer=layer3 --confirm=no --dataset_name=megadepth_sushi --suffix=stage_1 --valid_iter=1000 --enable_zoom=no --crop_cam=crop_center_and_resize --out_dir=./my_out/cotr --resume True

# # debug(proven)
# python train_cotr.py --scene_file sample_data/jsons/debug_megadepth.json --info_level=rgbd --use_ram=no --use_cc=no --batch_size=24 --learning_rate=1e-4 --lr_backbone=0 --max_iter=300000 --workers=8 --cycle_consis=yes --bidirectional=yes --position_embedding=lin_sine --layer=layer3 --confirm=no --dataset_name=megadepth_sushi --suffix=stage_1 --valid_iter=1000 --enable_zoom=no --crop_cam=crop_center_and_resize --out_dir=./my_out/cotr --resume False

#
python train_cotr.py --scene_file sample_data/jsons/debug_megadepth.json --info_level=rgbd --use_ram=yes --use_cc=no --batch_size=24 --learning_rate=1e-4 --lr_backbone=0 --max_iter=300000 --workers=8 --cycle_consis=yes --bidirectional=yes --position_embedding=lin_sine --layer=layer3 --confirm=no --dataset_name=megadepth_sushi --suffix=stage_1 --valid_iter=1000 --enable_zoom=no --crop_cam=crop_center_and_resize --out_dir=./my_out/cotr --resume=yes


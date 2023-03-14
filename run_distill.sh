#!/bin/bash

# val loss doesn't go down
# python distill.py --load_weights="default" --scene_file sample_data/jsons/debug_megadepth.json --info_level=rgbd --use_ram=yes --use_cc=no --batch_size=24 --learning_rate=1e-4 --lr_backbone=0 --max_iter=300000 --workers=8 --cycle_consis=yes --bidirectional=yes --position_embedding=lin_sine --layer=layer3 --s_layer=layer2 --confirm=no --dataset_name=megadepth_sushi --suffix=distill_1 --valid_iter=1000 --enable_zoom=no --crop_cam=crop_center_and_resize --out_dir=./out --resume=no

#
# python distill.py --load_weights="default" --scene_file sample_data/jsons/200_megadepth.json --info_level=rgbd --use_ram=yes --use_cc=no --batch_size=24 --learning_rate=0 --lr_backbone=1e-5 --max_iter=300000 --workers=8 --cycle_consis=yes --bidirectional=yes --position_embedding=lin_sine --layer=layer3 --s_layer=layer2 --confirm=no --dataset_name=megadepth_sushi --suffix=distill_1 --valid_iter=1000 --enable_zoom=no --crop_cam=crop_center_and_resize --out_dir=./out --resume=no

# debug, profile
# kernprof -l  distill.py --load_t_weights="out/default" --load_s_weights="out/default" --scene_file sample_data/jsons/debug_megadepth.json --info_level=rgbd --use_ram=no --use_cc=no --batch_size=24 --learning_rate=1e-4 --max_iter=100000 --workers=8 --cycle_consis=yes --bidirectional=yes --position_embedding=lin_sine --layer=layer3 --s_layer=layer2 --confirm=no --dataset_name=megadepth_sushi --suffix=distill_1_debug --valid_iter=50 --enable_zoom=no --crop_cam=crop_center_and_resize --out_dir=./my_out --resume=no --cc_resume=no

# python -m line_profiler distill.py.lprof > distill_prof.txt

# # student : layer2
# python distill.py --load_t_weights="out/default" --load_s_weights="out/default" --scene_file sample_data/jsons/200_megadepth.json --info_level=rgbd --use_ram=no --use_cc=no --batch_size=24 --learning_rate=1e-4 --max_iter=100000 --workers=8 --cycle_consis=yes --bidirectional=yes --position_embedding=lin_sine --layer=layer3 --s_layer=layer2 --confirm=no --dataset_name=megadepth_sushi --suffix=distill_200 --valid_iter=50 --enable_zoom=no --crop_cam=crop_center_and_resize --out_dir=./my_out --resume=no --cc_resume=no

# student : layer1
python distill.py --load_t_weights="out/default" --load_s_weights="out/default" \
 --layer=layer3 --s_layer=layer1 \
 --scene_file sample_data/jsons/debug_megadepth.json \
 --suffix=distill_2_debug \
 --info_level=rgbd --use_ram=no --use_cc=no --batch_size=24 --learning_rate=1e-4 --max_iter=100000 --workers=8 --position_embedding=lin_sine \
 --confirm=no --dataset_name=megadepth_sushi \
 --valid_iter=50 --enable_zoom=no --crop_cam=crop_center_and_resize --out_dir=./my_out --resume=no --cc_resume=no

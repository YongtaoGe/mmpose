
#!/usr/bin/env bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/analysis/coords_get_flops.py configs/coord_pose/coco/res18_4_layers_deformable_decoder_coco_256x192_3x.py


#python ./tools/analysis/coords_get_flops.py configs/coord_hp_pose/coco/res18_simple_baseline_coco_256x192.py

#python ./tools/analysis/coords_get_flops.py configs/top_down/coord_pose/coco/res50_wo_fpn_coco_256x192.py

#python ./tools/analysis/coords_get_flops.py configs/top_down/coord_pose/coco/rsn18_coco_256x192.py




#python ./tools/analysis/coords_get_flops.py configs/top_down/coord_pose/coco/res18_4_layers_baseline_coco_256x192.py
#python ./tools/analysis/coords_get_flops.py configs/top_down/coord_pose/coco/hrnet_w32_3_layers_deformable_decoder_coco_256x192.py

#python ./tools/analysis/coords_get_flops.py configs/top_down/fc_pose/coco/fc_pose_res18_coco_256x192.py


#python ./tools/analysis/coords_get_flops.py configs/top_down/coord_pose/coco/res18_4_layers_standard_decoder_coco_256x192.py

#python ./tools/analysis/coords_get_flops.py configs/top_down/coord_pose/coco/res18_wo_fpn_coco_256x192.py

#python ./tools/analysis/coords_get_flops.py configs/top_down/coord_pose/coco/res18_coco_256x192.py




#python ./tools/analysis/coords_get_flops.py configs/top_down/coord_pose/coco/2xrsn18_coco_256x192.py

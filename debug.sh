
#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python mmpose/models/keypoint_heads/deformable_transformer.py
#python ./tools/analysis/coords_get_flops.py configs/top_down/coord_pose/coco/res18_wo_fpn_coco_256x192.py

#python ./tools/analysis/coords_get_flops.py configs/top_down/coord_pose/coco/res18_coco_256x192.py



#python ./tools/analysis/coords_get_flops.py configs/top_down/coord_pose/coco/rsn18_coco_256x192.py
#python ./tools/analysis/coords_get_flops.py configs/top_down/coord_pose/coco/2xrsn18_coco_256x192.py

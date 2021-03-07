#CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=11117 ./tools/dist_train.sh configs/top_down/coord_pose/coco/res18_wo_fpn_coco_256x192.py 4

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=11118 ./tools/dist_train.sh configs/top_down/coord_pose/coco/res18_wo_fpn_coco_256x192.py 1
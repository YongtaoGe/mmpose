#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=11128 ./tools/dist_test.sh \
#configs/coord_pose/coco_new/res18_4_feat_0_enc_6_dec_coco_256x192_3x.py \
#work_dirs/res18_4_feat_0_enc_6_dec_coco_256x192_3x/epoch_300.pth 8 \
#--eval mAP

#GPUS=8 CPUS_PER_TASK=2 \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=11128 ./tools/dist_test.sh \
#configs/coord_pose/coco_new/res18_3_feat_6_enc_6_dec_hp_coco_256x192_3x.py \
#work_dirs/res18_3_feat_6_enc_6_dec_hp_coco_256x192_3x/epoch_320.pth 1 \
#--eval mAP

#GPUS=8 CPUS_PER_TASK=2 \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=11128 ./tools/dist_test.sh \
#configs/coord_pose/coco_new/res50_3_feat_6_enc_6_dec_hp_coco_384x288_4x.py \
#work_dirs/res50_3_feat_6_enc_6_dec_hp_coco_384x288_4x/epoch_310.pth 8 \
#--eval mAP

#GPUS=8 CPUS_PER_TASK=2 \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=11128 ./tools/dist_test.sh \
#configs/coord_pose/coco_new/res101_3_feat_6_enc_6_dec_hp_coco_384x288_4x_test.py \
#work_dirs/res101_3_feat_6_enc_6_dec_hp_coco_384x288_4x/epoch_360.pth 8 \
#--eval mAP

#GPUS=8 CPUS_PER_TASK=2 \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=11128 ./tools/dist_test.sh \
#configs/coord_pose/coco_new/res50_3_feat_4_enc_6_dec_hp_coco_256x192_4x_test.py \
#work_dirs/res50_3_feat_4_enc_6_dec_hp_coco_256x192_4x/epoch_400.pth 8 \
#--eval mAP


GPUS=8 CPUS_PER_TASK=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=11129 ./tools/dist_test.sh \
configs/coord_pose/coco_new/res50_3_feat_6_enc_6_dec_hp_coco_384x288_4x_test.py \
work_dirs/res50_3_feat_6_enc_6_dec_hp_coco_384x288_4x/epoch_400.pth 8 \
--eval mAP
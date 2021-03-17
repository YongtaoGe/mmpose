#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=11128 ./tools/dist_test.sh \
#configs/coord_pose/coco_new/res18_4_feat_0_enc_6_dec_coco_256x192_3x.py \
#work_dirs/res18_4_feat_0_enc_6_dec_coco_256x192_3x/epoch_300.pth 8 \
#--eval mAP
GPUS=8 CPUS_PER_TASK=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=11128 ./tools/dist_test.sh \
configs/coord_pose/coco_new/res18_3_feat_6_enc_6_dec_hp_coco_256x192_3x.py \
work_dirs/res18_3_feat_6_enc_6_dec_hp_coco_256x192_3x/epoch_320.pth 1 \
--eval mAP

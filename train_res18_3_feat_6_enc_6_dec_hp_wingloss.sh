CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=11118 ./tools/dist_train.sh \
configs/coord_pose/coco_wingloss/res18_3_feat_6_enc_6_dec_hp_wingloss_coco_256x192_3x.py 8 \
#--resume-from ./work_dirs/res18_3_feat_6_enc_6_dec_hp_wingloss_coco_256x192_3x/epoch_100.pth
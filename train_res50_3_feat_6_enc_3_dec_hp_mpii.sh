CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=11118 ./tools/dist_train.sh \
configs/coord_pose/mpii/res50_3_feat_6_enc_3_dec_hp_mpii_256x256_3x.py 8 \
--resume-from ./work_dirs/res50_3_feat_6_enc_3_dec_hp_mpii_256x256_3x/epoch_20.pth
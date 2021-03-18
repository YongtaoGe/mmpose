
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=11128 ./tools/dist_test.sh \
#configs/coord_pose/mpii/res50_3_feat_6_enc_3_dec_hp_mpii_256x256_3x_test.py \
#work_dirs/res50_3_feat_6_enc_3_dec_hp_mpii_256x256_3x/epoch_320.pth 8 \
#--eval PCKh

GPUS=8 CPUS_PER_TASK=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=11128 ./tools/dist_test.sh \
configs/coord_pose/mpii/res101_3_feat_6_enc_3_dec_hp_mpii_256x256_3x_test.py \
work_dirs/res101_3_feat_6_enc_3_dec_hp_mpii_256x256_3x/epoch_320.pth 8 \
--eval PCKh
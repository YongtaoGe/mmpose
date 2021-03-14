
#!/usr/bin/env bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/analysis/coords_get_flops.py  configs/coord_pose/coco_new/res101_3_feat_4_enc_6_dec_coco_256x192_3x.py
#python ./tools/analysis/coords_get_flops.py  configs/coord_pose/coco_new/res50_4_feat_1_enc_6_dec_hp_coco_256x192_3x.py



#python ./tools/analysis/coords_get_flops.py configs/coord_pose/coco_new/res101_3_feat_4_enc_6_dec_coco_256x192_3x.py
#python ./tools/analysis/coords_get_flops.py configs/coord_pose/coco_new/res18_4_feat_0_enc_6_dec_coco_256x192_3x_fpn.py

#python ./tools/analysis/coords_get_flops.py configs/coord_pose/coco_new/res50_3_feat_4_enc_6_dec_coco_384x288_3x.py --shape 320 240

#python ./tools/analysis/coords_get_flops.py configs/coord_pose/coco_new/res18_4_feat_1_enc_6_dec_coco_384x288_3x.py --shape 384 288

#python ./tools/analysis/coords_get_flops.py configs/coord_pose/coco_new/res50_3_feat_4_enc_6_dec_coco_256x192_3x.py

#python ./tools/analysis/coords_get_flops.py configs/top_down/resnet/coco/res152_coco_256x192.py
#python ./tools/analysis/coords_get_flops.py configs/top_down/deeppose/coco/deeppose_res50_coco_256x192.py



#python ./tools/analysis/coords_get_flops.py configs/coord_pose/coco/res18_3_layers_6_encoder_deformable_wo_self_attn_decoder_coco_256x192_6x.py
#python ./tools/analysis/coords_get_flops.py configs/coord_pose/coco/res18_4_layers_deformable_share_query_decoder_coco_256x192_3x.py

#python ./tools/analysis/coords_get_flops.py configs/coord_pose/coco/res18_4_layers_deformable_decoder_coco_256x192_3x.py




#python ./tools/analysis/coords_get_flops.py configs/coord_pose/coco/res18_4_layers_deformable_decoder_coco_256x192_3x.py


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

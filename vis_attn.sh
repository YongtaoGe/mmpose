#!/usr/bin/env bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/analysis/vis_attn.py \
    configs/coord_pose/coco_new/res50_3_feat_6_enc_6_dec_hp_coco_384x288_4x.py \
    work_dirs/res50_3_feat_6_enc_6_dec_hp_coco_384x288_4x/epoch_400.pth \
    --img-root data/coco/val2017/ --json-file data/coco/annotations/person_keypoints_val2017.json \
    --out-img-root vis_attn_results


#python demo/top_down_img_demo.py \
#    configs/coord_pose/coco_new/res50_3_feat_6_enc_6_dec_hp_coco_384x288_4x.py \
#    work_dirs/res50_3_feat_6_enc_6_dec_hp_coco_384x288_4x/epoch_400.pth \
#    --img-root data/coco/test2017/ --json-file data/coco/person_detection_results/COCO_test-dev2017_detections_AP_H_609_person.json \
#    --out-img-root vis_results

#python demo/top_down_img_demo.py \
#    configs/coord_pose/coco_new/res50_3_feat_6_enc_6_dec_hp_coco_384x288_4x.py \
#    work_dirs/res50_3_feat_6_enc_6_dec_hp_coco_384x288_4x/epoch_400.pth \
#    --img-root tests/data/coco/ --json-file tests/data/coco/test_coco.json \
#    --out-img-root vis_results


#python demo/top_down_img_demo.py \
#    configs/coord_pose/coco_hrnet/hrnet_w32_3_feat_3_enc_3_dec_hp_coco_256x192_4x.py \
#    work_dirs/hrnet_w32_3_feat_3_enc_3_dec_hp_coco_256x192_4x/epoch_300.pth \
#    --img-root tests/data/coco/ --json-file tests/data/coco/test_coco.json \
#    --out-img-root vis_results
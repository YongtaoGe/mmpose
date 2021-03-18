#python tools/analysis/analyze_logs.py plot_curve \
#./work_dirs/res18_3_feat_6_enc_6_dec_coco_256x192_3x/20210311_173705.log.json \
#./work_dirs/res18_3_feat_6_enc_6_dec_hp_coco_256x192_3x/20210313_224724.log.json \
#--keys coord_acc \
#--out results1.pdf \
#--legend wo_aux_loss with_aux_loss \
#--style whitegrid

python tools/analysis/analyze_logs.py plot_curve \
./work_dirs/res18_3_feat_6_enc_6_dec_coco_256x192_3x/20210311_173705.log_1.json \
./work_dirs/res18_3_feat_6_enc_6_dec_hp_coco_256x192_3x/20210313_224724.log.json \
--keys coord_acc \
--out results3.pdf \
--legend wo_aux_loss with_aux_loss \
--style ticks
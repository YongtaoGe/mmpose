PHASE='train'
# MODE='transformed'
MODE='original'
SHOW_INTERVAL=1000
OUTPUT_DIR='./vis_bedlam_1'
CONFIG=configs/controlpose/rtmpose-l_8xb256-420e_bedlam-256x192.py

python tools/misc/browse_dataset.py \
${CONFIG} \
--output-dir ${OUTPUT_DIR} \
--not-show \
--phase ${PHASE} \
--mode ${MODE} \
--show-interval ${SHOW_INTERVAL}
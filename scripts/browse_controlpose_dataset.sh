PHASE='train'
# MODE='transformed'
MODE='original'
SHOW_INTERVAL=10
OUTPUT_DIR='./vis_controlpose_pseudo'
CONFIG=configs/controlpose/rtmpose-l_8xb256-420e_controlpose-256x192.py

python tools/misc/browse_dataset.py \
${CONFIG} \
--output-dir ${OUTPUT_DIR} \
--not-show \
--phase ${PHASE} \
--mode ${MODE} \
--show-interval ${SHOW_INTERVAL}

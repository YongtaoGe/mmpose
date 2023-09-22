# export NCCL_P2P_DISABLE=1
# ./tools/dist_train.sh \
# configs/controlpose/rtmpose-l_8xb256-40e_controlpose-256x192.py \
# 8 \


./tools/dist_train.sh \
configs/controlpose/rtmpose-l_8xb256-40e_controlpose_coco_0.1-256x192.py \
8 \

# ${CONFIG_FILE} ${GPU_NUM} [PY_ARGS]

# export NCCL_P2P_DISABLE=1
t ./tools/dist_train.sh \
configs/controlpose/rtmpose-l_8xb256-420e_coco-256x192.py \
2 \

# ${CONFIG_FILE} ${GPU_NUM} [PY_ARGS]

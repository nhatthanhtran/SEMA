# 1x schedule
torchrun --nproc_per_node 8 tools/train.py ./configs/sema/sema_mrcnn_1x.py --launcher="pytorch"

# 3x schedule
# torchrun --nproc_per_node 8 tools/train.py ./configs/sema/sema_mrcnn_3x.py --launcher="pytorch"


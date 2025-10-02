# multi-gpu training
torchrun --nproc_per_node 8 tools/train.py ./segmentation/configs/sema/sema_upernet.py  --launcher="pytorch"
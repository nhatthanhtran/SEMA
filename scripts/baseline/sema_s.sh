python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg cfgs/baseline/sema_s.yaml --data-path <imagenet-path> --output ./result/ --amp 

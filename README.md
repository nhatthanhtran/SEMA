# SEMA: a Scable and Efficient Mamba like Attention via Token Localization and Averaging

This repo contains the official PyTorch code for **Scable and Efficient Mamba like Attention (SEMA)**




## Abstract

Attention is the critical component of a transformer. Yet the quadratic computational complexity of vanilla full attention in the input size and the inability of its linear attention variant to focus have been  challenges for computer vision tasks. 
We provide a mathematical definition of generalized attention and formulate both vanilla softmax attention and linear attention within the general framework. We prove that generalized attention disperses, that is, as the number of keys tends to infinity, the query assigns equal weights to all keys. Motivated by the dispersion property and recent development of Mamba form of attention, we design Scalable and Efficient Mamba like Attention (SEMA) which utilizes token localization to avoid dispersion and maintain focusing, complemented by theoretically consistent  arithmetic averaging to capture global aspect of attention. We support our approach on Imagenet-1k where classification results show that SEMA is a scalable and effective alternative beyond linear attention, outperforming recent vision Mamba models on increasingly larger scales of images at similar model parameter sizes. 

## Instruction
We provide the entire code base which is similar to Swin and MILA.
The models folder includes SEMA model. We provide the scripts to run the training 
for the T/S model. In addition, downstream folder include all of the files to train
downstream tasks on COCO and ADE20K.  

## Dependencies

- Python 3.9
- PyTorch == 1.11.0
- torchvision == 0.12.0
- numpy
- timm == 0.4.12
- yacs



Please visit the official website: https://www.image-net.org/ to retrieve the dataset
The ImageNet dataset should be prepared as follows:

```
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img2.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img3.jpeg
    │   └── ...
    ├── class2
    │   ├── img4.jpeg
    │   └── ...
    └── ...
```


## Model Training and Inference

- Evaluate one model on ImageNet:

```
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg <path-to-config-file> --data-path <imagenet-path> --output <output-path> --eval --resume <path-to-pretrained-weights>
```

- To train `SEMA-T/S` on ImageNet from scratch, run:

```
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg <path-to-config-file> --data-path <imagenet-path> --output <output-path> --amp
```

## Acknowledgements

This code is developed on the top of [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [FLatten Transformer](https://github.com/LeapLabTHU/FLatten-Transformer) and [Agent Attention](https://github.com/LeapLabTHU/Agent-Attention), 
and [Mamba Inspire Linear Attention](https://github.com/LeapLabTHU/MLLA) 

## Citation


## Contact


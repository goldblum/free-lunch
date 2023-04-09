# Model Selection with Neural Networks

The code in this folder trains or loads two neural networks, and then learns a convex combination of their logits which prefers the simpler model insofar as it fits the training data.  This procedure is a very simple prototype for how a single learner can perform well on both small and large datasets (e.g. CIFAR-10 and ImageNet) where single neural network architectures previously performed well on at most only small or large but not both.

## Train Neural Networks:

```bash
python3 train.py [directory containing datasets] --imagenet_resize --arch googlenet --seed 0 --dataset_name CIFAR10 --checkpoint_filename googlenet_CIFAR10.pth.tar
python3 train.py [directory containing datasets] --imagenet_resize --arch vit_b_16 --seed 0 --dataset_name CIFAR10 --checkpoint_filename vit_b_16_CIFAR10.pth.tar
```

## Train Convex Combination:

Here, the `--weight_decay` argument specifies the coefficient for the penalty on the parameter controlling the convex combination between logits.  A higher penalty coefficient corresponds to a more severe preference for the the smaller model (`--arch2`).

```bash
python3 train_tradeoff.py [directory containing datasets] --seed 0 --imagenet_resize --dataset_name CIFAR10 --arch1 vit_b_16 --arch2 googlenet --resume1 ./checkpoint/vit_b_16_CIFAR10.pth.tar --resume2 ./checkpoint/googlenet_CIFAR10.pth.tar --epochs 10 --weight_decay [penalty coefficient] 
python3 train_tradeoff.py [directory containing ImageNet] --seed 0 --imagenet_resize --dataset_name ImageNet --arch1 vit_b_16 --arch2 googlenet --weight_decay [penalty coefficient] --pretrained --epochs 10
```

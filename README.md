# Open-LACU
Code for reproducing results for the article 'UNIFIED MACHINE LEARNING: OPEN-SET LEARNING WITH AUGMENTED CATEGORY BY EXPLOITING UNLABELLED DATA (OPEN-LACU) ---- https://arxiv.org/abs/2002.01368


To run code 
  1. Decompress data folders
  2. Requirements: Pytorch and Cuda enabled device
  3. Run:
  
python3 MarginGAN_main.py     --dataset cifar10     --train-subdir train+val     --eval-subdir test     
   --batch-size 128     --labeled-batch-size 31     --arch cifar_shakeshake26     --consistency-type mse     
   --consistency-rampup 5     --consistency 100.0     --logit-distance-cost 0.01     --weight-decay 2e-4     
   --lr-rampup 0     --lr 0.05     --nesterov True     --labels data-local/labels/cifar10/4000_balanced_labels/00.txt      
   --epochs 181     --lr-rampdown-epochs 210     --ema-decay 0.97     --generated-batch-size 32     --out_dataset cifar100

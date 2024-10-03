Our method mainly refers to https://github.com/baofff/U-ViT
Pretrained model from U-ViT.

1. Progressive Traning.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup accelerate launch --multi_gpu --num_processes 8 --main_process_port 20639 --mixed_precision fp16 train_c10_multi_step.py --config=configs/cifar10_uvit_small_multi_step.py > output/cifar10_uvit_small_multi_step.txt & 


2. Traning only for 2:4 Sparsity
CUDA_VISIBLE_DEVICES=0,3,2,1 nohup accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_c10_mask_pretrained.py --config=configs/cifar10_uvit_small_linear_pretrained.py > output/cifar10_uvit_small_linear_pretrained.txt & 

3. Inference only 2:4 Sparsity
CUDA_VISIBLE_DEVICES=3,2,1,0 nohup accelerate launch --multi_gpu --num_processes 4 --main_process_port 20653 --mixed_precision fp16 eval.py --config=configs/cifar10_uvit_small_linear_pretrained.py --nnet_path=./workdir/cifar10_uvit_small_linear_pretrained/default/ckpts/500000.ckpt/nnet_ema.pth --config.sample.path=sample/eval_c10_500000_linear_pretrained > output/eval_c10_500000_linear_pretrained.txt &


4. Traning for other scale inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup accelerate launch --multi_gpu --num_processes 8 --main_process_port 20639 --mixed_precision fp16 train_c10_mask_step.py --config=configs/cifar10_uvit_small_linear_step7_8.py > output/cifar10_uvit_small_linear_step7_8.txt & 

5. Inference 7:8 scale 
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval.py --config=configs/cifar10_uvit_small_linear_eval_multi_78.py --nnet_path=./workdir/cifar10_uvit_small_linear_dense8_8/default/ckpts/500000.ckpt/nnet_ema.pth --config.sample.path=sample/eval_c10_linear_multi_78 > output/eval_c10_linear_multi_78.txt &

6. Transfer learn sparse mask
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --multi_gpu --num_processes 4 --main_process_port 20632 --mixed_precision fp16 train_c10_ls2.py --config=configs/cifar10_uvit_small_linear_ls.py > output/cifar10_uvit_small_linear_ls.txt &

7. Transfer learn sparse mask, loss
CUDA_VISIBLE_DEVICES=4,5 nohup accelerate launch --multi_gpu --num_processes 2 --main_process_port 20631 --mixed_precision fp16 train_c10_lsm3.py --config=configs/cifar10_uvit_small_linear_lsm3.py > output/cifar10_uvit_small_linear_lsm3.txt & 

DDPM UNet refers to  https://github.com/Hramchenko/diffusion_distiller
6. Training for UNet
CUDA_VISIBLE_DEVICES=1,5,6,9 nohup accelerate launch --multi_gpu --num_processes 4 --main_process_port 20638 --mixed_precision fp16 train_ddpm.py --config=configs/cifar10_unet_ddpm3.py > output/cifar10_unet_ddpm3.txt & 

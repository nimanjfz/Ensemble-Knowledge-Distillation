# Ensemble Knowledge Distillation
This repo contains implementation code for the ensemble knowledge distillation project
- For distilling knowledge from the ensemble of teachers to the student network
```bash
python main.py --run 2 --gpu_id 0 --lr 0.1 --batch_size 128  --teachers [\'resnet34\',\'resnet50\'] --student resnet18 --out_dims [10000,5000,1000,500,10572] --d_lr 1e-3 --fc_out 1 --pool_out avg --loss ce --adv 1 --gamma [1,1,1,1,1] --eta [1,1,1,1,1] --name casia_contrastive --out_layer [-1]

```

## Environment
Python 3.6+

PyTorch 0.40+

Numpy 1.12+ 

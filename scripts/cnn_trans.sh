CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset_name TESS \
  --lr 0.00001 \
  --epoch 20 \
  --weight_decay 0.005 \
  --batch_size 16 \
  --save_epoch 10 \
  --num_layers 4

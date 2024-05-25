srun --job-name=trans --mem=8G --gres=gpu:1 --cpus-per-task=8 python train.py \
  --dataset_name ESD \
  --lr 0.00001 \
  --epoch 20 \
  --weight_decay 0.005 \
  --batch_size 16 \
  --save_epoch 10 \
  --num_layers 4

CUDA_VISIBLE_DEVICES=2 python eval.py \
  --dataset_name EMO-DB_de \
  --ckpt_path ckpts/cnn-transformer-mix-50.pt \
  --batch_size 16 \
  --num_layers 4

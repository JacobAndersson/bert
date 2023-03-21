python3 train.py \
  --device="cuda:0" \
  --batch_size=128 \
  --grad_accumulation_steps=32 \
  --n_layers=12 \
  --n_heads=12 \
  --model_dim=768 \
  --ff_dim=3072 \
  --testing_interval=20 \
  --bias=False

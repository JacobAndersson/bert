python3 train.py \
  --device="cuda:0" \
  --batch_size=128 \
  --grad_accumulation_steps=32 \
  --n_layers=12 \
  --n_heads=12 \
  --d_model=768 \
  --d_ff=3072 \
  --bias=False

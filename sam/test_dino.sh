export CUDA_VISIBLE_DEVICES=7
python grounding_dino_zy.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint ./ckpt/groundingdino_swint_ogc.pth \
  --input_image_dir "/share/sd/pipe/debug"  \
  --output_dir "/share/sd/pipe/dino_output" \
  --box_threshold 0.1 \
  --text_threshold 0.1 \
  --text_prompt "text, like text" \
  --device "cuda"
# python grounding_dino_demo.py \
#   --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
#   --grounded_checkpoint ./ckpt/groundingdino_swint_ogc.pth \
#   --input_image assets/demo1.jpg \
#   --output_dir "outputs" \
#   --box_threshold 0.3 \
#   --text_threshold 0.25 \
#   --text_prompt "bear" \
#   --device "cuda"
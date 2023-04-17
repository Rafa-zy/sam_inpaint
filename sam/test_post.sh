export CUDA_VISIBLE_DEVICES=7
python auto_post_process_zy.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint ./ckpt/groundingdino_swint_ogc.pth \
  --sam_checkpoint ./ckpt/sam_vit_h_4b8939.pth \
  --input_image "Nothing.jpg" \
  --output_dir "/share/sd/sam/Grounded-Segment-Anything-main/outputs" \
  --openai_key "A Faster VIP OPENAI-APIID :)" \
  --box_threshold 0.3 \
  --text_threshold 0.2 \
  --iou_threshold 0.5 \
  --inpaint_mode "first" \
  --device "cuda"

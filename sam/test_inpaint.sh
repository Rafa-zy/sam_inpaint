CUDA_VISIBLE_DEVICES=0
python grounded_sam_inpainting_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint ./ckpt/groundingdino_swint_ogc.pth \
  --sam_checkpoint ./ckpt/sam_vit_h_4b8939.pth \
  --input_image assets/inpaint_demo.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --det_prompt "bench" \
  --inpaint_prompt "Nothing" \
  --device "cuda"
# python grounded_sam_inpainting_demo.py \
#   --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
#   --grounded_checkpoint ./ckpt/groundingdino_swint_ogc.pth \
#   --sam_checkpoint ./ckpt/sam_vit_h_4b8939.pth \
#   --input_image assets/inpaint_demo.jpg \
#   --output_dir "outputs" \
#   --box_threshold 0.3 \
#   --text_threshold 0.25 \
#   --det_prompt "bench" \
#   --inpaint_prompt "A sofa, high quality, detailed" \
#   --device "cuda"
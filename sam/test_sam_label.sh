export CUDA_VISIBLE_DEVICES=7
python automatic_label_zy.py \
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
# python automatic_label_demo.py \
#   --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
#   --grounded_checkpoint ./ckpt/groundingdino_swint_ogc.pth \
#   --sam_checkpoint ./ckpt/sam_vit_h_4b8939.pth \
#   --input_image "/share/sd/hf/eval_system/rlhf_output/db_0330_text_epoch_450_train_2/ Soft Brown Modern Dream House Banner_0.jpg" \
#   --output_dir "outputs" \
#   --openai_key "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" \
#   --box_threshold 0.3 \
#   --text_threshold 0.2 \
#   --iou_threshold 0.5 \
#   --device "cuda"
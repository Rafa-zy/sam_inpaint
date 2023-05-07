# README

We aim at constructing dataset for multi-modal instruction tuning, including the original image, the related caption (text) prompt), the cropped image with single object, and the detected bounding box, the segmentation mask, and the image without objects, together with the related caption after removing the objects (background text prompt).

```
{
"image":,
"image_prompt":
"image_wo_object":,
"image_wo_object_prompt":
"object_image_prompt": [obj_name1, ... ],
"object_image": [obj_1, ... ],
"bounding_box": [b1, b2, ...]
}
```

## Environment

Please follow the original repo of Grounded-SAM to manage your environemnt for training and inference.

## How to run the code

```
bash test_sam_label.sh
```

add customization for coco dataset
```
bash test_sam_label_coco.sh
# you should change the following codes
# 1. local coco data path
# 2. local save path
# 3. OPENAI API key
```



remember to fill in the VIP-OpenAI API ID to speedup the process

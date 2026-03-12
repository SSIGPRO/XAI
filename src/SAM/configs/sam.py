from pathlib import Path

#--------------------------------
# SAM2 model
#--------------------------------
# Install: pip install git+https://github.com/facebookresearch/sam2.git
# Download checkpoints from: https://github.com/facebookresearch/sam2?tab=readme-ov-file#model-description

sam2_checkpoint = '/srv/newpenny/XAI/models/sam2/sam2.1_hiera_large.pt'
sam2_model_cfg  = 'configs/sam2.1/sam2.1_hiera_l.yaml'   # resolved by sam2 internally

#--------------------------------
# Grounding DINO (for text-prompted segmentation)
#--------------------------------
# Install: pip install groundingdino-py
# Checkpoint: https://github.com/IDEA-Research/GroundingDINO#luggage-checkpoints
gdino_checkpoint = '/srv/newpenny/XAI/models/grounding_dino/groundingdino_swint_ogc.pth'
gdino_model_cfg  = '/srv/newpenny/conda/envs/xai-venv/lib/python3.12/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py'
gdino_box_thresh = 0.3
gdino_text_thresh = 0.25

#--------------------------------
# Automatic Mask Generator params
#--------------------------------
amg_kwargs = dict(
    points_per_side        = 16,
    pred_iou_thresh        = 0.5,
    stability_score_thresh = 0.80,
    crop_n_layers          = 0,
    min_mask_region_area   = 25,    # ignore tiny blobs (pixels)
)

#--------------------------------
# Input resolution for SAM2 (upscale small images before inference)
#--------------------------------
sam_input_size = 1024   # SAM2 is designed for ~1024x1024; upscale from 224x224

#--------------------------------
# Concept storage
#--------------------------------
top_k_masks  = 10      # keep at most this many masks per image (sorted by stability score)
concepts_path = Path('../../data') / 'sam_concepts'

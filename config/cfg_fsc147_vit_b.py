data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
data_aug_max_size = 1333
data_aug_scales2_resize = [400, 500, 600]
data_aug_scales2_crop = [384, 600]
data_aug_scale_overlap = None
batch_size = 4
modelname = 'groundingdino'

# ===========================================================================
# Upgrade 5: Backbone Selection
# Options (all ~88M params, fits on 3090 24GB):
#   "swin_B_384_22k"                                        — Original (default)
#   "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k"    — Swin-V2-B (+0.9% ImageNet)
#   "convnextv2_base.fcmae_ft_in22k_in1k_384"               — ConvNeXt-V2-B (+1.1% ImageNet)
#   "focalnet_base_lrf.in1k"                                 — FocalNet-B (best for detection)
#   "eva02_large_patch14_448"                                 — EVA-02-L (304M, needs A100+)
# ===========================================================================
backbone = "swin_B_384_22k"
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
# --- Upgrade 2: Query count (original: 900, reduced: 400) ---
num_queries = 400
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = 'standard'
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = 'relu'
dec_pred_bbox_embed_share = True
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
embed_init_tgt = True
dn_labelbook_size = 91
max_text_len = 256
text_encoder_type = "bert-base-uncased"
use_text_enhancer = True
use_fusion_layer = True
use_checkpoint = True
use_transformer_ckpt = True
use_text_cross_attention = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1
sub_sentence_present = True
max_labels = 90                               # pos + neg
lr = 0.0001                                   # base learning rate
backbone_freeze_keywords = None               # only for gdino backbone
freeze_keywords = ['backbone.0', 'bert']      # for whole model, e.g. ['backbone.0', 'bert'] for freeze visual encoder and text encoder
lr_backbone = 1e-05                           # specific learning rate
lr_backbone_names = ['backbone.0', 'bert']
lr_linear_proj_mult = 1e-05
lr_linear_proj_names = ['ref_point_head', 'sampling_offsets']
weight_decay = 0.0001
param_dict_type = 'ddetr_in_mmdet'
ddetr_lr_param = False
# --- Upgrade 4: When cosine_lr=True, consider increasing epochs (original: 30) ---
epochs = 30
lr_drop = 10                         # Only used when cosine_lr=False
save_checkpoint_interval = 10
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = False
lr_drop_list = [10, 20]
frozen_weights = None
dilation = False
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
random_refpoints_xy = False
fix_refpoints_hw = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False
use_deformable_box_attn = False
box_attn_type = 'roi_align'
dec_layer_number = None
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = False
add_pos_value = False
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
# --- Upgrade 2: Must match num_queries (original: 900) ---
num_select = 400
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
aux_loss = True
# --- Upgrade 1: GIoU loss (set both to 0.0 to disable, 2.0 to enable) ---
set_cost_class = 5.0
set_cost_bbox = 1.0
set_cost_giou = 2.0                 # Original: 0.0
cls_loss_coef = 5.0
bbox_loss_coef = 1.0
giou_loss_coef = 2.0                # Original: 0.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
mask_loss_coef = 1.0
dice_loss_coef = 1.0
focal_alpha = 0.25
focal_gamma = 2.0
decoder_sa_type = 'sa'
matcher_type = 'HungarianMatcher'
decoder_module_seq = ['sa', 'ca', 'ffn']
nms_iou_threshold = -1
dec_pred_class_embed_share = True
match_unstable_error = True
use_detached_boxes_dec_out = False
dn_scalar = 100

# --- Upgrade 3: torch.compile ---
use_torch_compile = False

# --- Upgrade 4: Cosine LR schedule ---
cosine_lr = False
cosine_lr_min = 1e-6
cosine_warmup_epochs = 2

# --- Upgrade 6: Density map head ---
use_density_head = False
density_loss_coef = 0.5
density_sigma = 3.0

# --- Upgrade 7: Multi-scale TTA ---
use_multiscale_tta = False
tta_scales = [0.75, 1.0, 1.25]
tta_nms_threshold = 0.5

box_threshold = 0.23
text_threshold = 0
use_coco_eval = False
label_list = ['alcohol bottle', 'baguette roll', 'ball', 'banana', 'bead', 'bee', 'birthday candle', 'biscuit', 'boat', 'bottle', 'bowl', 'box', 'bread roll', 'brick', 'buffalo', 'bun', 'calamari ring', 'can', 'candle', 'cap', 'car', 'cartridge', 'cassette', 'cement bag', 'cereal', 'chewing gum piece', 'chopstick', 'clam', 'coffee bean', 'coin', 'cotton ball', 'cow', 'crane', 'crayon', 'croissant', 'crow', 'cup', 'cupcake', 'cupcake holder', 'fish', 'gemstone', 'go game piece', 'goat', 'goldfish snack', 'goose', 'ice cream', 'ice cream cone', 'instant noodle', 'jade stone', 'jeans', 'kidney bean', 'kitchen towel', 'lighter', 'lipstick', 'm&m piece', 'macaron', 'match', 'meat skewer', 'mini blind', 'mosaic tile', 'naan bread', 'nail', 'nut', 'onion ring', 'orange', 'pearl', 'pen', 'pencil', 'penguin', 'pepper', 'person', 'pigeon', 'plate', 'polka dot tile', 'potato', 'rice bag', 'roof tile', 'screw', 'shoe', 'spoon', 'spring roll', 'stair', 'stapler pin', 'straw', 'supermarket shelf', 'swan', 'tomato', 'watermelon', 'window', 'zebra']
val_label_list = ['ant', 'bird', 'book', 'bottle cap', 'bullet', 'camel', 'chair', 'chicken wing', 'donut', 'donut holder', 'flamingo', 'flower', 'flower pot', 'grape', 'horse', 'kiwi', 'milk carton', 'oyster', 'oyster shell', 'package of fresh cut fruit', 'peach', 'pill', 'polka dot', 'prawn cracker', 'sausage', 'seagull', 'shallot', 'shirt', 'skateboard', 'toilet paper roll']

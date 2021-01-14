# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# cascade_mask_rcnn_r2_101_fpn_20e_coco.py -> 4*2 0.01
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 19])
total_epochs = 20

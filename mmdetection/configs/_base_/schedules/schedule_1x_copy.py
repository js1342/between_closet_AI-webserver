# optimizer
# htc -> 0.001, 2*2
# mask_rcnn -> 0.005 2*2

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='Adam', lr=0.0005, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12

config = {
    'train_dir': 'archive/cropped_train_images/cropped_train_images',
    'batch_size': 2,
    'model_name': 'efficientnet_b6',
    'img_size': 512,
    'num_class': 15587,
    'embedding_size': 128,
    'n_splits': 5,
    'lr': 0.0001,
    'weight_decay': 0.000001,
    'T_max': 500,
    'min_lr': 0.00001,
    'epochs': 4,
    'device': 'cuda',
    #'model_path': 'runs/weight/EffNetB0_fold_1_loss_14.35.pt'
    'model_path': None
}
class DefaultConfigs(object):
    #1.string parameters
    # train_data = "/home/imdl/DataSets/vol2/TCT/PATCH/853-512/inf/TP/classification/single-core/train/"
    # val_data = "/home/imdl/DataSets/vol2/TCT/PATCH/853-512/inf/TP/classification/single-core/val/"
    # train_data = val_data = test_data = '/home/imdl/DataSets/vol2/TCT/PATCH/853-512/inf/TP/classification/multicore.csv'
    # train_data = val_data ='/home/imdl/DataSets/vol2/CAMELYON16/training/patch/inf/train_tumor_t_normal_tn100.csv'
    # test_data = '/home/imdl/DataSets/vol2/CAMELYON16/training/patch/inf/val.csv'

    train_data = val_data = '/home/imdl/DataSets/vol2/CAMELYON16/training/patch/inf/512/train2.csv'
    test_data = '/home/imdl/DataSets/vol2/CAMELYON16/training/patch/inf/512/val2.csv'
    # train_data = val_data = '/home/imdl/DataSets/vol2/Mitosis/classification/val.csv' tumor_256_
    # test_data = "/home/imdl/DataSets/vol2/TCT/PATCH/853-512/inf/TP/classification/all/test/"
    # test_data = "/home/imdl/DataSets/vol2/TCT/PATCH/853-512/inf/TP/853-512-train/hsil"
    # test_data = "/home/imdl/DataSets/vol2/TCT/PATCH/853-512/inf/TP/classification/single-core/wsi_roi/201904-neg"
    # test_data = "/home/imdl/DataSets/vol2/TCT/PATCH/853-512/inf/TP/classification/single-core/wsi_roi/201907-neg"

    # test_data = "/home/imdl/DataSets/vol2/TCT/PATCH/853-512/inf/TP/hsil/multicore_hsil"
    # test_data = "/home/imdl/DataSets/vol2/TCT/PATCH/neg/202008-201912neg/201912NegPatchRoi50-hsil/sample1000/multicore_hsil"

    # test_data = "/home/imdl/DataSets/vol2/TCT/PATCH/853-512/inf/TP/classification/wsi_roi/POS-single_core_hsil-853"
    # test_data = "/home/imdl/DataSets/vol2/TCT/PATCH/853-512/inf/TP/classification/single-core/wsi_roi/NEG-single_core_hsil-201907"
    # test_data = "/home/imdl/DataSets/vol2/TCT/PATCH/853-512/inf/TP/classification/single-core/test/"
    # test_data = "/home/imdl/DataSets/vol2/TCT/PATCH/853-512/inf/TP/classification/wsi_roi/20200323-pos"
    # model_name = "resnet50"
    # model_name = "resnet18"
    model_name = 'deeplab'
    # model_name = 'res2net50'
    # backbone = 'backbone'

    weights = "/home/imdl/DataSets/vol2/workspace/pytorch-image-classification/CAMELYON16_checkpoints/"
    best_models = weights + "res2net50/512patch_train/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "1"
    augmen_level = "medium"  # "medium","light","hard","hard2"

    #2.numeric parameters
    epochs = 100
    batch_size = 8
    img_height = 512
    img_weight = img_height

    num_classes = 2
    seed = 888
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4

config = DefaultConfigs()

import torch


configurations = {
    1: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/media/data3/tyingf', # the parent root where your train/val/test data are stored
        MODEL_ROOT = './checkpoints', # the root to buffer your checkpoints
        TRAIN_DATASET = 'casia_ori', # ['casia_ori', 'ms1m_ibug'] the path of your train dataset
        BACKBONE_RESUME_ROOT = None, # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = None, # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Softmax', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 256,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 100, # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [20, 28, 32], # epoch stages to decay learning rate

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0,6], # specify your GPU ids
        PIN_MEMORY = True,
        NUM_WORKERS = 8,
        BETA = 0.2,
        GAMMA = 1.0,
        CON_A = 0.87, # 0.87 for CASIA-WebFace; 0.90 for MS1M-IBUG
        CON_B = 1.20, # 1.20 for CASIA-WebFace; 1.20 for MS1M-IBUG
),
}

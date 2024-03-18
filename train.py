import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from config import configurations
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from util.utils import make_weights_for_balanced_classes, get_val_data_cfp_fp, get_val_data_lfw, separate_irse_bn_paras, separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy
from util.data import FaceDataset
from tqdm import tqdm
from loss import FocalLoss, FCGLoss, cal_loss_p

import os


if __name__ == '__main__':

    #======= hyperparameters & data loaders =======#
    cfg = configurations[1]

    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
    MODEL_ROOT = cfg['MODEL_ROOT'] # the root to buffer your checkpoints
    TRAIN_DATASET = cfg['TRAIN_DATASET']
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    HEAD_NAME = cfg['HEAD_NAME'] # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME'] # support: ['Focal', 'Softmax']

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST'] # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR'] # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES'] # epoch stages to decay learning rate

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID'] # specify your GPU ids
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']
    BETA = cfg['BETA']
    GAMMA = cfg['GAMMA']
    CON_A = cfg['CON_A']
    CON_B = cfg['CON_B']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)

    if not os.path.exists(MODEL_ROOT):
        os.mkdir(MODEL_ROOT)

    train_transform = transforms.Compose([
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])
    
    dataset_train = FaceDataset(os.path.join(DATA_ROOT, TRAIN_DATASET, 'imgs'), train_transform)

    # create a weighted random sampler to process imbalanced data
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size = BATCH_SIZE, sampler = sampler, pin_memory = PIN_MEMORY,
        num_workers = NUM_WORKERS, drop_last = DROP_LAST
    )
    '''
    train_loader = DataLoaderX(
        dataset_train, batch_size = BATCH_SIZE, sampler = sampler, pin_memory = PIN_MEMORY,
        num_workers = NUM_WORKERS, drop_last = DROP_LAST
    )
    '''
    NUM_CLASS = len(train_loader.dataset.classes)
    print("Number of Training Classes: {}".format(NUM_CLASS))

    cfp_fp, cfp_fp_issame = get_val_data_cfp_fp(os.path.join(DATA_ROOT, 'validate_dataset', 'cfp_align_112'))
    lfw, lfw_issame = get_val_data_lfw(os.path.join(DATA_ROOT, 'validate_dataset', 'lfw_align_112'))
    
    #======= model & loss & optimizer =======#
    BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE), 
                     'ResNet_101': ResNet_101(INPUT_SIZE), 
                     'ResNet_152': ResNet_152(INPUT_SIZE),
                     'IR_50': IR_50(INPUT_SIZE), 
                     'IR_101': IR_101(INPUT_SIZE), 
                     'IR_152': IR_152(INPUT_SIZE),
                     'IR_SE_50': IR_SE_50(INPUT_SIZE), 
                     'IR_SE_101': IR_SE_101(INPUT_SIZE), 
                     'IR_SE_152': IR_SE_152(INPUT_SIZE)}
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    print("=" * 60)
    print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

    HEAD_DICT = {'ArcFace': ArcFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'CosFace': CosFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'SphereFace': SphereFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'Am_softmax': Am_softmax(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID)}
    HEAD = HEAD_DICT[HEAD_NAME]
    print("=" * 60)
    print(HEAD)
    print("{} Head Generated".format(HEAD_NAME))
    print("=" * 60)

    LOSS_DICT = {'Focal': FocalLoss(), 
                 'Softmax': nn.CrossEntropyLoss()}
    
    criterion_xent = LOSS_DICT[LOSS_NAME]
    criterion_fcg = FCGLoss(features_size = EMBEDDING_SIZE, total_classes = NUM_CLASS, device_id = GPU_ID, a=CON_A, b=CON_B)
    
    print("=" * 60)
    print(criterion_xent)
    print("{} Loss Generated".format(LOSS_NAME))
    print("=" * 60)

    if BACKBONE_NAME.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_resnet_bn_paras(HEAD)
    
    optimizer_model = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY}, {'params': backbone_paras_only_bn}], lr = LR, momentum = MOMENTUM)
    optimizer_fcg = optim.SGD(criterion_fcg.parameters(), lr = 0.1)

    print("=" * 60)
    print(optimizer_model)
    print("Optimizer Generated")
    print("=" * 60)

    # optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT and HEAD_RESUME_ROOT:
        print("=" * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT) and os.path.isfile(HEAD_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
            print("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
            HEAD.load_state_dict(torch.load(HEAD_RESUME_ROOT))
        else:
            print("No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT))
        print("=" * 60)

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)


    #======= train & validation & save checkpoint =======#
    DISP_FREQ = len(train_loader) // 10 # frequency to display training loss & acc

    NUM_EPOCH_WARM_UP = NUM_EPOCH // 25  # use the first 1/25 epochs to warm up
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up
    batch = 0  # batch index

    for epoch in range(NUM_EPOCH): # start training process
        
        if epoch == STAGES[0]: # adjust LR for each training stage after warm up, you can also choose to adjust LR manually (with slight modification) once plaueau observed
            schedule_lr(optimizer_model)
        if epoch == STAGES[1]:
            schedule_lr(optimizer_model)
        if epoch == STAGES[2]:
            schedule_lr(optimizer_model)

        BACKBONE.train()  # set to training mode
        HEAD.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for inputs, labels, yaws in tqdm(iter(train_loader)):

            if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (batch + 1 <= NUM_BATCH_WARM_UP): # adjust LR for each training batch during warm up
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, optimizer_model)

            # compute output
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            yaws = yaws.to(DEVICE).float()
            
            features = BACKBONE(inputs) # [B, 512]
            outputs, centers = HEAD(features, labels) # centers: [classes, 512]
            loss_y = criterion_xent(outputs, labels)

            # divide frontal and profile faces
            f_mask = yaws < 15.0
            p_mask = ~f_mask
            f_fea, p_fea = features[f_mask], features[p_mask]
            f_labels, p_labels = labels[f_mask], labels[p_mask]
  
            # calculate loss
            loss_f, f_centers, mode_yi = criterion_fcg(f_fea, f_labels)
            loss_p = cal_loss_p(p_fea, p_labels, f_centers, GPU_ID, NUM_CLASS, mode_yi, CON_A, CON_B)
            loss_center = (- torch.cosine_similarity(f_centers, centers)).mean()
            
            loss_f = BETA * loss_f
            loss_p = BETA * loss_p
            loss_center = GAMMA * loss_center

            loss = loss_y + loss_f + loss_p + loss_center
            
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk = (1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))
            top5.update(prec5.data.item(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer_model.zero_grad()
            optimizer_fcg.zero_grad()

            loss.backward()

            optimizer_model.step()
            for param in criterion_fcg.parameters():
                param.grad.data *= (1. / BETA)

            optimizer_fcg.step()
            # dispaly training loss & acc every DISP_FREQ
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                print("=" * 60)
                print('Epoch {}/{} Batch {}/{}\t'
                      'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch + 1, NUM_EPOCH, batch + 1, len(train_loader) * NUM_EPOCH, loss = losses, top1 = top1, top5 = top5))
                print("=" * 60)

            batch += 1 # batch index

        # training statistics per epoch (buffer for visualization)
        epoch_loss = losses.avg
        epoch_acc = top1.avg

        print("=" * 60)
        print('Epoch: {}/{}\t'
              'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            epoch + 1, NUM_EPOCH, loss = losses, top1 = top1, top5 = top5))
        print("=" * 60)
        with open('training.txt', 'a') as f:
            f.write('Epoch: {}/{} loss: {:.4f}, Prec@1: {:.3f}, Prec@5f: {:.5f}\n'.format(epoch + 1, NUM_EPOCH, losses.avg, top1.avg, top5.avg))
        f.close()
        
        print("=" * 60)
        accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame)
        accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_fp, cfp_fp_issame)
        
        print("Epoch {}/{}, Evaluation: LFW Acc: {}".format(epoch + 1, NUM_EPOCH, accuracy_lfw))
        with open('validate_lfw.txt', 'a') as f:
            f.write("Epoch {}/{}, Evaluation: LFW Acc: {}, best_threshold_lfw: {}\n".format(epoch + 1, NUM_EPOCH, accuracy_lfw, best_threshold_lfw))
        f.close()

        print("Epoch {}/{}, Evaluation: CFP_FP Acc: {}".format(epoch + 1, NUM_EPOCH, accuracy_cfp_fp))
        with open('validate_cfp_fp.txt', 'a') as f:
            f.write("Epoch {}/{}, Evaluation: CFP_FP Acc: {}, best_threshold_cfp_fp: {}\n".format(epoch + 1, NUM_EPOCH, accuracy_cfp_fp, best_threshold_cfp_fp))
        f.close()
        
        print("=" * 60)
        
        # save checkpoints per epoch
        if MULTI_GPU:
            torch.save(BACKBONE.module.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch, get_time())))
            torch.save(HEAD.state_dict(), os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch, get_time())))
        else:
            torch.save(BACKBONE.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch, get_time())))
            torch.save(HEAD.state_dict(), os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch, get_time())))

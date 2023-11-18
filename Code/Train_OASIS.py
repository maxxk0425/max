# python imports
import os
import glob
import warnings
# external imports
import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam,SGD
import torch.utils.data as Data
# internal imports
import sys
sys.path.append("/home/bb/temprun/MeetMorph-main/")
from Model import losses_g3
from Model.config import args
from Model.datagenerators import DatasetFromFolder3D_Fixed, DatasetFromFolder3D_Moving
from Model.model_g3 import  U_Network3, SpatialTransformer, SpatialTransformer_nearest
from skimage import transform
from Model.Deformable_skip_learner import Deformable_Skip_Learner
from torchvision import transforms
from torch import nn
import nibabel as nib

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def resize_image_itk_Linear(itkimage, newSize, resamplemethod=sitk.sitkLinear):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled

def make_dirs():
    if not os.path.exists(args.model_dir_M2F):
        os.makedirs(args.model_dir_M2F)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir_M2F):
        os.makedirs(args.result_dir_M2F)

def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir_M2F, name))

def train():
    # gpu
    make_dirs()
    device = torch.device("cuda:3")
    # log
    log_name = str(args.n_iter) + "_" + str(args.lr) + "_" + str(args.alpha)
    print("log_name: ", log_name)
    f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")

    #  fixed
    f_img = sitk.ReadImage("/home/bb/temprun/MeetMorph-main/data/OASIS/OASIS_128/fixed/S279.delineation.skullstripped.nii.gz")
    f_img1 = resize_image_itk_Linear(f_img, (128, 128, 128), resamplemethod=sitk.sitkLinear)
    f_img2 = resize_image_itk_Linear(f_img, (64, 64, 64), resamplemethod=sitk.sitkLinear)
    f_img3 = resize_image_itk_Linear(f_img, (32, 32, 32), resamplemethod=sitk.sitkLinear)
    f_img4 = resize_image_itk_Linear(f_img, (16, 16, 16), resamplemethod=sitk.sitkLinear)

    f_img1 = sitk.GetArrayFromImage(f_img1)[np.newaxis, np.newaxis, ...]
    f_img2 = sitk.GetArrayFromImage(f_img2)[np.newaxis, np.newaxis, ...]
    f_img3 = sitk.GetArrayFromImage(f_img3)[np.newaxis, np.newaxis, ...]
    f_img4 = sitk.GetArrayFromImage(f_img4)[np.newaxis, np.newaxis, ...]

    vol_size1 = f_img1.shape[2:]
    vol_size2 = f_img2.shape[2:]
    vol_size3 = f_img3.shape[2:]
    vol_size4 = f_img4.shape[2:]

    #print(len(vol_size1))

    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]

    UNet1 = U_Network3(len(vol_size3), nf_enc, nf_dec).to(device)
    Fusion = Deformable_Skip_Learner([3, 3, 3, 3]).to(device)

    Fusion.train()
    UNet1.train()

    STN1 = SpatialTransformer(vol_size1).to(device)
    STN2 = SpatialTransformer(vol_size2).to(device)
    STN3 = SpatialTransformer(vol_size3).to(device)
    STN4 = SpatialTransformer(vol_size4).to(device)

    STN1.train()
    STN2.train()
    STN3.train()
    STN4.train()

    # parameters
    print("UNet1: ", count_parameters(UNet1))
    print("STN1: ", count_parameters(STN1))

    # Set optimizer and losses
    # opt1 = Adam(UNet1.parameters(), lr=args.lr)
    opt1 = SGD(UNet1.parameters(), lr=0.1, momentum=0.9)
    sim_loss_fn = losses_g3.ncc_loss if args.sim_loss == "ncc" else losses_g3.mse_loss
    grad_loss_fn = losses_g3.gradient_loss

    train_dir_fixed = glob.glob(os.path.join("/home/bb/temprun/MeetMorph-main/data/OASIS/OASIS_128/OASIS", '*.nii.gz'))
    train_dir_moving = glob.glob(os.path.join("/home/bb/temprun/MeetMorph-main/data/OASIS/OASIS_128/OASIS", '*.nii.gz'))

    # Training loop.
    for i in range(1, 200000 + 1):
        # get fixed
        dataloader_Fixed = DatasetFromFolder3D_Fixed(train_dir_fixed)
        data_Fixed = Data.DataLoader(dataloader_Fixed, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                     drop_last=True)

        # get moving
        dataloader_Moving = DatasetFromFolder3D_Moving(train_dir_moving)
        data_Moving = Data.DataLoader(dataloader_Moving, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                      drop_last=True)

        # Generate the moving and fixed images and convert them to tensors.
        mov_img, mov_img_2, mov_img_4, mov_img_8 = next(data_Moving.__iter__())
        fix_img, fix_img_2, fix_img_4, fix_img_8 = next(data_Fixed.__iter__())

        input_moving = mov_img
        input_moving = input_moving.to(device).float()

        input_fixed = fix_img
        input_fixed = input_fixed.to(device).float()

        input_moving_2 = mov_img_2
        input_moving_2 = input_moving_2.to(device).float()

        input_fixed_2 = fix_img_2
        input_fixed_2 = input_fixed_2.to(device).float()

        input_moving_4 = mov_img_4
        input_moving_4 = input_moving_4.to(device).float()

        input_fixed_4 = fix_img_4
        input_fixed_4 = input_fixed_4.to(device).float()

        Upsample = torch.nn.Upsample(scale_factor=2, mode="trilinear")

        ''' Registration lv1 '''
        flow4 = UNet1(input_moving_4, input_fixed_4)
        M2F4_1 = STN3(input_moving_4, flow4)

        flow_meta4 = UNet1(input_fixed_4, M2F4_1)
        v4 = flow4 - flow_meta4

        #######################################
        #######################################

        M2F4_1_ = STN3(input_moving_4, v4)

        flow4_ = UNet1(M2F4_1_, input_fixed_4)
        M2F4_1_ = STN3(M2F4_1_, flow4_)

        flow_meta4_ = UNet1(input_fixed_4, M2F4_1_)
        v4_ = flow4_ - flow_meta4_

        #######################################
        #######################################
        v4_up = Upsample(v4)
        v4__up = Upsample(v4_)

        V4_UP = v4_up + v4__up

        v4_add = v4 + v4_
        M2F4_v4_add = STN3(input_moving_4, v4_add)

        loss_s_v4_add = 0 * grad_loss_fn(v4_add)
        loss_ncc_v4 = 0.1 * sim_loss_fn(M2F4_v4_add, input_fixed_4)
        loss_v4 = loss_s_v4_add + loss_ncc_v4

        ''' Registration lv2 '''

        M2F2_1 = STN2(input_moving_2, V4_UP)

        flow2 = UNet1(M2F2_1, input_fixed_2)
        M2F2_2 = STN2(M2F2_1, flow2)

        flow_meta2 = UNet1(input_fixed_2, M2F2_2)
        v2 = flow2 - flow_meta2

        #######################################
        #######################################
        M2F2_2_ = STN2(M2F2_1, v2)

        flow2_ = UNet1(M2F2_2_, input_fixed_2)
        M2F2_2_ = STN2(M2F2_2_, flow2_)

        flow_meta2_ = UNet1(input_fixed_2, M2F2_2_)
        v2_ = flow2_ - flow_meta2_
        v2_add = v2 + v2_
        #######################################
        #######################################
        v2__up = Upsample(v2_)
        v2_up = Upsample(v2)
        v4_up_up = Upsample(V4_UP)

        v24 =  v4_up_up + v2_up + v2__up

        Moved1 = STN1(input_moving, v24)

        v2_add = v2 + v2_ + V4_UP
        M2F2_v2_add = STN2(input_moving_2, v2_add)

        loss_s_v2_add = 0 * grad_loss_fn(v2_add)
        loss_ncc_v2 = 0.1 * sim_loss_fn(M2F2_v2_add, input_fixed_2)
        loss_v2 = loss_ncc_v2 + loss_s_v2_add

        ''' Registration lv3 '''
        flow3 = UNet1(Moved1, input_fixed)
        M2F_3 = STN1(Moved1, flow3)

        flow_meta3 = UNet1(input_fixed, M2F_3)
        v3 = flow3 - flow_meta3
        #######################################
        #######################################
        M2F_3_ = STN1(Moved1, v3)

        flow3_ = UNet1(M2F_3_, input_fixed)
        M2F_3_ = STN1(M2F_3_, flow3_)

        flow_meta3_ = UNet1(input_fixed, M2F_3_)
        v3_ = flow3_ - flow_meta3_

        #######################################
        #######################################
        VV = v3 + v3_ + v24
        Movedx = STN1(input_moving, VV)

        # cauclate  generate loss
        loss_ncc_VV = sim_loss_fn(Movedx, input_fixed)
        loss_ncc = loss_ncc_VV
        # loss_s = 1 * grad_loss_fn(v3) + 1* grad_loss_fn(v3_) + 1 * grad_loss_fn(v24)
        loss_s = 1 * grad_loss_fn(v3) + 1 * grad_loss_fn(v3_) + 1 * grad_loss_fn(v24)
        # loss_s =  4 * grad_loss_fn(v3) + 4 * grad_loss_fn(v3_) + 2 * grad_loss_fn(v2) + 2 * grad_loss_fn(v2_) + 1 * grad_loss_fn(v4) + 1 * grad_loss_fn(v4_)
        loss = loss_ncc + loss_s

        opt1.zero_grad()
        loss.backward()
        opt1.step()
        print("i: %d  loss: %f  simVV: %f  grad: %f" % (i, loss.item(), loss_ncc_VV.item(), loss_s.item()))

        if i % 1000 == 0:
            # Save model checkpoint
            save_file_name1 = os.path.join("/home/bb/temprun/MeetMorph-main/CVPR23/ours/Train_OASIS/model_dir_OASIS_128_FBI", '%d_Unet1.pth' % i)
            torch.save(UNet1.state_dict(), save_file_name1)

            print("warped images have saved.")
    f.close()

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()

# python imports
import os
import glob
import warnings
import time
# external imports
import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
# internal imports
import sys
sys.path.append("/home/bb/temprun/MeetMorph-main/")
from Model import losses
from Model.config import args
from Model.datagenerators import DatasetFromFolder3D_Fixed, DatasetFromFolder3D_Moving
from Model.model import U_Network1, U_Network3, SpatialTransformer, SpatialTransformer_nearest, jacobian_determinant, VecInt, CompositionTransform, generate_grid
from skimage import transform
from Model.Deformable_skip_learner import Deformable_Skip_Learner
from torchvision import transforms
from torch import nn
import nibabel as nib
import torch.nn.functional as F
import matplotlib.pyplot as plt
def make_dirs():
    if not os.path.exists(args.Test_result):
        os.makedirs(args.result_dir)

def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.Test_result, name))

def resize_image_itk_Nearest(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
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

class SpatialTransform_flow(nn.Module):
    """
    N-D Spatial Transformer
    """
    def __init__(self, size):
        super(SpatialTransform_flow, self).__init__()

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid, persistent=False)

    def forward(self, src, flow, mode='bilinear'):

        new_locs = flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return F.grid_sample(grid, new_locs, align_corners=True, mode=mode)

def create_grid(size, path):
    num1, num2, num3 = (size[0] + 10) // 10, (size[1] + 10) // 10, (size[2] + 10) // 10
    x, y = np.meshgrid(np.linspace(-2, 2, num1), np.linspace(-2, 2, num2))

    plt.figure(figsize=((size[0] + 10) / 100.0, (size[1] + 10) / 100.0))
    plt.plot(x, y, color="black")
    plt.plot(x.transpose(), y.transpose(), color="black")
    plt.axis('off')

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(path)



def compute_label_dice(gt, pred):

    #cls_lst = [1,2,3,4]
    cls_lst = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31, 32, 33, 34,35]
    # cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
    #            63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
    #            163, 164, 165, 166]
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)

# @torchsnooper.snoop()
def test():
    make_dirs()
    device = torch.device("cuda:0")
    #  fixed
    f_img = sitk.ReadImage("/home/bb/temprun/MeetMorph-main/data/OASIS/OASIS_128/fixed/S279.delineation.skullstripped.nii.gz")
    f_img_4 = sitk.ReadImage(
        "/home/bb/temprun/MeetMorph-main/data/OASIS/OASIS_32/S279.delineation.skullstripped.nii.gz")
    f_img1 = resize_image_itk_Linear(f_img, (128, 128, 128), resamplemethod=sitk.sitkLinear)
    f_img2 = resize_image_itk_Linear(f_img, (64, 64, 64), resamplemethod=sitk.sitkLinear)
    f_img4 = resize_image_itk_Linear(f_img, (32, 32, 32), resamplemethod=sitk.sitkLinear)
    f_img8 = resize_image_itk_Linear(f_img, (16, 16, 16), resamplemethod=sitk.sitkLinear)

    f_img1 = sitk.GetArrayFromImage(f_img1)[np.newaxis, np.newaxis, ...]
    f_img2 = sitk.GetArrayFromImage(f_img2)[np.newaxis, np.newaxis, ...]
    f_img4 = sitk.GetArrayFromImage(f_img4)[np.newaxis, np.newaxis, ...]
    f_img8 = sitk.GetArrayFromImage(f_img8)[np.newaxis, np.newaxis, ...]

    vol_size1 = f_img1.shape[2:]
    vol_size2 = f_img2.shape[2:]
    vol_size4 = f_img4.shape[2:]
    vol_size8 = f_img8.shape[2:]

    Template = "/home/bb/temprun/MeetMorph-main/data/OASIS/OASIS_128/S01/S01.delineation.skullstripped.nii.gz"
    Template_label_OASIS = "/home/bb/temprun/MeetMorph-main/data/OASIS/OASIS_128/S01/S01.delineation.structure.label.nii.gz"

    atlas = sitk.ReadImage(Template)
    atlas1 = resize_image_itk_Linear(atlas,(128, 128, 128), resamplemethod=sitk.sitkLinear)
    atlas2 = resize_image_itk_Linear(atlas, (64, 64, 64), resamplemethod=sitk.sitkLinear)
    atlas4 = resize_image_itk_Linear(atlas, (32, 32, 32), resamplemethod=sitk.sitkLinear)
    atlas8 = resize_image_itk_Linear(atlas, (16, 16, 16), resamplemethod=sitk.sitkLinear)

    atlas_t = sitk.GetArrayFromImage(atlas1)[np.newaxis, np.newaxis, ...]
    atlas1 = sitk.GetArrayFromImage(atlas1)[np.newaxis, np.newaxis, ...]
    atlas2 = sitk.GetArrayFromImage(atlas2)[np.newaxis, np.newaxis, ...]
    atlas4 = sitk.GetArrayFromImage(atlas4)[np.newaxis, np.newaxis, ...]
    atlas8 = sitk.GetArrayFromImage(atlas8)[np.newaxis, np.newaxis, ...]

    # set up atlas tensor
    atlas1 = torch.from_numpy(atlas1).to(device).float()
    atlas2 = torch.from_numpy(atlas2).to(device).float()
    atlas4 = torch.from_numpy(atlas4).to(device).float()
    atlas8 = torch.from_numpy(atlas8).to(device).float()

    # Test file and anatomical labels we want to evaluate
    test_file_lst = glob.glob(os.path.join("/home/bb/temprun/MeetMorph-main/data/OASIS/OASIS_128/S_01_img", "S354.delineation.skullstripped.nii.gz"))
    print("The number of test data: ", len(test_file_lst))

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    # Set up model
    UNet1 = U_Network3(len(vol_size4), nf_enc, nf_dec).to(device)
    Diffeomorphic = VecInt([128, 128, 128], 7).to(device)
    Compos_flow = CompositionTransform().to(device)
    imgshape = (128,128,128)
    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    n_iter = 1000
    for i in range(1, n_iter + 1):
        # u1 = 57300 * i
        # u1 = 44000
        u1 =  17000 * i
        # u1 =  70000 + 1000 * i
        u1 = os.path.join("/home/bb/temprun/MeetMorph-main/CVPR23/ours/Train_OASIS/model_OASIS_adma_lambda_1", str(u1) + "_Unet1.pth") # model_dir_squeeze_CNN_lv3  # model_dir_OASIS_para_025 # model_dir_OASIS_FBI
        UNet1.load_state_dict(torch.load(u1))
        print(u1)

        STN_img = SpatialTransformer(vol_size1).to(device)
        STN_label = SpatialTransformer(vol_size1, mode="nearest").to(device)
        STN_img2 = SpatialTransformer(vol_size2).to(device)
        STN_label2 = SpatialTransformer(vol_size2, mode="nearest").to(device)
        STN_img4 = SpatialTransformer(vol_size4).to(device)
        STN_label4 = SpatialTransformer(vol_size4, mode="nearest").to(device)
        STN_img8 = SpatialTransformer(vol_size8).to(device)
        STN_label8 = SpatialTransformer(vol_size8, mode="nearest").to(device)
        STN_flow_4 = SpatialTransform_flow(vol_size4).to(device)
        UNet1.eval()

        STN_img.eval()
        STN_label.eval()

        STN_img2.eval()
        STN_label2.eval()

        STN_img4.eval()
        STN_label4.eval()

        STN_img8.eval()
        STN_label8.eval()

        DSC = []
        JAC = []
        Times = []
        # Atlas label
        Atlas_label = sitk.ReadImage(Template_label_OASIS)

        Atlas_label = resize_image_itk_Nearest(Atlas_label, (128, 128, 128), resamplemethod=sitk.sitkNearestNeighbor)
        # Atlas_label2 = resize_image_itk_Nearest(Atlas_label, (64, 64, 64), resamplemethod=sitk.sitkNearestNeighbor)
        # Atlas_label4 = resize_image_itk_Nearest(Atlas_label, (32, 32, 32), resamplemethod=sitk.sitkNearestNeighbor)

        Atlas_label = sitk.GetArrayFromImage(Atlas_label)[np.newaxis, np.newaxis, ...]
        Atlas_label = torch.from_numpy(Atlas_label).to(device).float()
        # Atlas_label2 = sitk.GetArrayFromImage(Atlas_label)[np.newaxis, np.newaxis, ...]
        # Atlas_label2 = torch.from_numpy(Atlas_label).to(device).float()
        # Atlas_label4 = sitk.GetArrayFromImage(Atlas_label)[np.newaxis, np.newaxis, ...]
        # Atlas_label4 = torch.from_numpy(Atlas_label).to(device).float()
        def map2sample(flow):

            size_tensor = flow.size()
            sample = generate_grid(size_tensor[1:-1])
            sample = torch.from_numpy(sample).unsqueeze(0)
            grid = flow + sample
            grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (size_tensor[3] - 1) * 2
            grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (size_tensor[2] - 1) * 2
            grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (size_tensor[1] - 1) * 2
            return grid

        for file in test_file_lst:
            name = os.path.split(file)[1]


            # 读入moving图像
            input_moving = sitk.ReadImage(file)
            input_moving = resize_image_itk_Linear(input_moving, (128, 128, 128), resamplemethod=sitk.sitkLinear)
            input_moving_2 = resize_image_itk_Linear(input_moving, (64, 64, 64), resamplemethod=sitk.sitkLinear)
            input_moving_4 = resize_image_itk_Linear(input_moving, (32, 32, 32), resamplemethod=sitk.sitkLinear)
            input_moving_8 = resize_image_itk_Linear(input_moving, (16, 16, 16), resamplemethod=sitk.sitkLinear)

            input_moving = sitk.GetArrayFromImage(input_moving)[np.newaxis, np.newaxis, ...]
            input_moving = torch.from_numpy(input_moving).to(device).float()
            input_moving_2 = sitk.GetArrayFromImage(input_moving_2)[np.newaxis, np.newaxis, ...]
            input_moving_2 = torch.from_numpy(input_moving_2).to(device).float()
            input_moving_4 = sitk.GetArrayFromImage(input_moving_4)[np.newaxis, np.newaxis, ...]
            input_moving_4 = torch.from_numpy(input_moving_4).to(device).float()

            input_moving_8 = sitk.GetArrayFromImage(input_moving_8)[np.newaxis, np.newaxis, ...]
            input_moving_8 = torch.from_numpy(input_moving_8).to(device).float()

            # 读入moving图像对应的label
            label_file = glob.glob(os.path.join("/home/bb/temprun/MeetMorph-main/data/OASIS/OASIS_128/S_01_label", name[:4] + "*"))[0]
            input_label = sitk.ReadImage(label_file)

            input_label = resize_image_itk_Linear(input_label, (128, 128, 128), resamplemethod=sitk.sitkNearestNeighbor)
            input_label2 = resize_image_itk_Linear(input_label, (64, 64, 64), resamplemethod=sitk.sitkNearestNeighbor)
            input_label4 = resize_image_itk_Linear(input_label, (32, 32, 32), resamplemethod=sitk.sitkNearestNeighbor)

            input_label = sitk.GetArrayFromImage(input_label)[np.newaxis, np.newaxis, ...]
            input_label = torch.from_numpy(input_label).to(device).float()

            input_label2 = sitk.GetArrayFromImage(input_label2)[np.newaxis, np.newaxis, ...]
            input_label2 = torch.from_numpy(input_label2).to(device).float()

            input_label4 = sitk.GetArrayFromImage(input_label4)[np.newaxis, np.newaxis, ...]
            input_label4 = torch.from_numpy(input_label4).to(device).float()

            ############## TO achieve registration #####

            Upsample = torch.nn.Upsample(scale_factor=2, mode="trilinear")

            ''' Registration lv1 '''
            start_time = time.time()
            flow4 = UNet1(input_moving_4, atlas4)
            flow4 = flow4.detach()

            M2F4_1 = STN_img4(input_moving_4, flow4)
            M2F4_1 = M2F4_1.detach()

            flow_meta4 = UNet1(atlas4, M2F4_1)
            flow_meta4 = flow_meta4.detach()

            v4 = flow4 - flow_meta4


            #######################################
            #######################################
            M2F4_1_ = STN_img4(input_moving_4, v4)

            flow4_ = UNet1(M2F4_1_, atlas4)
            M2F4_1_ = STN_img4(M2F4_1_, flow4_)

            flow4_ = flow4_.detach()
            M2F4_1_ = M2F4_1_.detach()

            flow_meta4_ = UNet1(atlas4, M2F4_1_)
            flow_meta4_ = flow_meta4_.detach()

            v4_ = flow4_ - flow_meta4_
            # v4_flow = map2sample(v4_)
            # v4_flow = STN_flow_4(input_moving_4, v4_)
            # v4 = STN_flow_4(v4)
            # save_image(v4_flow.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img_4,
            #            "/home/bb/temprun/MeetMorph-main/Result_test/v4.nii.gz")
            # hhh



            v4_2 = Upsample(v4)
            v4_2_2 = Upsample(v4_2)

            v4_2_ = Upsample(v4_)
            v4_2_2_ = Upsample(v4_2_)



            # save_image(v4_2_2.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img,
            #            "/home/bb/temprun/MeetMorph-main/Result_test/v4_2_2.nii.gz")
            # save_image((v4_2_2_).permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img,
            #            "/home/bb/temprun/MeetMorph-main/Result_test/v4_2_2_.nii.gz")

            #######################################
            #######################################
            v4_up = Upsample(v4)
            v4__up = Upsample(v4_)

            V4_UP = v4_up + v4__up

            ''' Registration lv2 '''

            M2F2_1 = STN_img2(input_moving_2, V4_UP)

            flow2 = UNet1(M2F2_1, atlas2)
            M2F2_2 = STN_img2(M2F2_1, flow2)

            flow2 = flow2.detach()
            M2F2_2 = M2F2_2.detach()

            flow_meta2 = UNet1(atlas2, M2F2_2)
            flow_meta2 = flow_meta2.detach()

            v2 = flow2 - flow_meta2
            #######################################
            #######################################
            M2F2_2_ = STN_img2(M2F2_1, v2)

            flow2_ = UNet1(M2F2_2_, atlas2)
            M2F2_2_ = STN_img2(M2F2_2_, flow2_)

            flow2_ = flow2_.detach()
            M2F2_2_ = M2F2_2_.detach()

            flow_meta2_ = UNet1(atlas2, M2F2_2_)
            flow_meta2_ = flow_meta2_.detach()

            v2_ = flow2_ - flow_meta2_

            v2_2 = Upsample(v2)

            v2_2_ = Upsample(v2_)

            # save_image(v2_2.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img,
            #            "/home/bb/temprun/MeetMorph-main/Result_test/v2_2.nii.gz")
            # save_image((v2_2_).permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img,
            #            "/home/bb/temprun/MeetMorph-main/Result_test/v2_2_.nii.gz")
            #######################################
            #######################################

            v2__up = Upsample(v2_)
            v2_up = Upsample(v2)
            v4_up_up = Upsample(V4_UP)

            v24 = v4_up_up + v2_up + v2__up

            Moved1 = STN_img(input_moving, v24)


            ''' Registration lv3 '''
            flow3 = UNet1(Moved1, atlas1)
            flow3 = flow3.detach()

            M2F_3 = STN_img(Moved1, flow3)
            M2F_3 = M2F_3.detach()

            flow_meta3 = UNet1(atlas1, M2F_3)
            flow_meta3 = flow_meta3.detach()

            v3 = flow3 - flow_meta3


            #######################################
            #######################################
            M2F_3_ = STN_img(Moved1, v3)

            flow3_ = UNet1(M2F_3_, atlas1)
            flow3_ = flow3_.detach()

            M2F_3_ = STN_img(M2F_3_, flow3_)

            flow_meta3_ = UNet1(atlas1, M2F_3_)
            flow_meta3_ = flow_meta3_.detach()

            vx = flow3_ - flow_meta3_
            # save_image(v3.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img,
            #            "/home/bb/temprun/MeetMorph-main/Result_test/v1.nii.gz")
            # save_image((vx).permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img,
            #            "/home/bb/temprun/MeetMorph-main/Result_test/v1_.nii.gz")
            #######################################
            #######################################
            VV = (v3 + vx + v24)
            # save_image((VV).permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img,
            #            "/home/bb/temprun/MeetMorph-main/Result_test/VV.nii.gz")
            # # save_image(VV.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img,
            # #            "/home/bb/temprun/MeetMorph-main/Result_test/VV_test.nii.gz")
            # hhhh
            # atlas_t = torch.tensor(atlas_t).cuda()
            # save_image(v24.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img,
            #            "/home/bb/temprun/MeetMorph-main/Result_test/V24.nii.gz")
            # save_image(v3.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img,
            #            "/home/bb/temprun/MeetMorph-main/Result_test/v3.nii.gz")
            # save_image(vx.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img,
            #            "/home/bb/temprun/MeetMorph-main/Result_test/vx.nii.gz")

            F_label = STN_label(input_label, VV)
            # F_img = STN_img(input_moving, VV)
            # save_image(F_img, f_img, "/home/bb/temprun/MeetMorph-main/Result_test/F_img.nii.gz")
            # save_image(input_moving, f_img, "/home/bb/temprun/MeetMorph-main/Result_test/input_moving.nii.gz")
            # save_image(atlas1, f_img, "/home/bb/temprun/MeetMorph-main/Result_test/input_fixed.nii.gz")
            #
            # hhh
            end_time = time.time()
            times = end_time - start_time
            # print("times:", times)
            Times.append(times)

            Atlas_label = torch.tensor(Atlas_label).cuda().data.cpu().numpy()

            # DSC
            dice = compute_label_dice(Atlas_label, F_label[0, 0, ...].cpu().detach().numpy())
            DSC.append(dice)

            # Jacc
            jac_det = np.sum(jacobian_determinant(VV.permute(0, 2, 3, 4, 1).squeeze(0).detach().cpu()) <= 0)
            tar = input_moving.detach().cpu().numpy()[0, 0, :, :, :]
            jac_neg_per = jac_det / np.prod(tar.shape)
            JAC.append(jac_neg_per)

            # print("dice: ", dice)
            # print("JAC",jac_neg_per)
        print("mean(DSC): ", np.mean(DSC), "   std(DSC): ", np.std(DSC))
        print("mean(JAC): ", np.mean(JAC), "   std(JAC): ", np.std(JAC))
        print("mean(Times): ", np.mean(Times), "   std(JAC): ", np.std(Times))

if __name__ == "__main__":
    test()

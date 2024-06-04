# This is the code to evaluate the performances, 
# in terms of BPP, mIoU and FID of the proposed SPIC framework

import os
import subprocess
import metrics_functions as mfunc
import numpy as np

##################
#### Parameters
##################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, required=True, help="folder of images to evaluate,"
                                                                  "containing samples and labels")
args = parser.parse_args()

result_path = args.result_dir
internimage_path = "/home/Users/dqy/Projects/InternImage/"


##################
#### Main
##################
if __name__ == '__main__':
    ##### evaluate the BPP for the coarse
    print("Evaluating BPP - coarse image by BPG")
    BPP_coarse = mfunc.calculate_bpp(os.path.join(result_path, "compressed"), ext=".bpg")
    
    ##### evaluate the BPP for the input SSM
    print("Evaluating BPP - SSM image by FLIF")
    mfunc.flif_compression(f"{result_path}/labels", f"{result_path}/labels")
    BPP_SSM = mfunc.calculate_bpp(os.path.join(result_path, "labels"), ext=".flif")
    
    ##### generate the SSM with internimage
    print("Evaluating mIOU - SS by InternImage")
    command = (f"cd {internimage_path} && python3 segmentation/image_demo.py "
               f"{result_path}/samples "
               f"--palette cityscapes "
               f"--opacity 1 "
               f"segmentation/configs/cityscapes/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.py "
               f"segmentation/checkpoint_dir/seg/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.pth "
               f"--out {result_path}/sem_generated")
    mfunc.run_command_with_conda_env("internimage", command)
    
    ##### evaluate the mIoU
    print("Evaluating mIOU")
    true_SSM = mfunc.get_semantic_maps(os.path.join(result_path, "labels/"), interimage=False)
    generated_SSM = mfunc.get_semantic_maps(os.path.join(result_path, "sem_generated/"), interimage=True)
    mIoU = mfunc.calculate_mIoU(true_SSM, generated_SSM)
    
    ##### evaluate the FID
    print("Evaluating FID")
    true_img_path = os.path.join(result_path, "images/")
    sampled_img_path = os.path.join(result_path, "samples/")
    fid = mfunc.calculate_FID(true_img_path, sampled_img_path)
    
    ##### print the results
    print(f"BPP: {round(BPP_coarse+BPP_SSM,4)}")
    print(f"BPP(SSM): {round(BPP_SSM,4)}")
    print(f"BPP(Com): {round(BPP_coarse,4)}")
    print(f"mIoU: {round(mIoU,4)}")
    print(f"FID: {round(fid,4)}")
    

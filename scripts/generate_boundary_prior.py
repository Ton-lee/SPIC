"""对于Cityscapes数据集，从json标注文件获取目标边界"""


#!/usr/bin/python
#
# Converts the polygonal annotations of the Cityscapes dataset
# to images, where pixel values encode the ground truth classes and the
# individual instance of that classes.
#
# The Cityscapes downloads already include such images
#   a) *color.png             : the class is encoded by its color
#   b) *labelIds.png          : the class is encoded by its ID
#   c) *instanceIds.png       : the class and the instance are encoded by an instance ID
# 
# With this tool, you can generate option
#   d) *instanceTrainIds.png  : the class and the instance are encoded by an instance training ID
# This encoding might come handy for training purposes. You can use
# the file labes.py to define the training IDs that suit your needs.
# Note however, that once you submit or evaluate results, the regular
# IDs are needed.
#
# Please refer to 'json2instanceImg.py' for an explanation of instance IDs.
#
# Uses the converter tool in 'json2instanceImg.py'
# Uses the mapping defined in 'labels.py'
#

# python imports
from __future__ import print_function, absolute_import, division
import os, glob, sys
sys.path.append("/home/Users/dqy/Dataset/Cityscapes/cityscapesScripts/")

# cityscapes imports
from cityscapesscripts.helpers.csHelpers import printError
from cityscapesscripts.preparation.json2instanceImg import json2instanceImg, json2boundary_prior
import cv2
resize = True
size1 = (1024, 512)
size2 = (512, 256)

# The main method
def main():
    save_root = "/home/Users/dqy/Dataset/Cityscapes/boundary_compressed(prior)/"
    # Where to look for Cityscapes
    if 'CITYSCAPES_DATASET' in os.environ:
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
    else:
        cityscapesPath = "/home/Users/dqy/Dataset/Cityscapes/"
    # how to search for all ground truth
    searchFine   = os.path.join( cityscapesPath , "gtFine"   , "*" , "*" , "*_gt*_polygons.json" )
    searchCoarse = os.path.join( cityscapesPath , "gtCoarse" , "*" , "*" , "*_gt*_polygons.json" )

    # search files
    filesFine = glob.glob( searchFine )
    filesFine.sort()
    filesCoarse = glob.glob( searchCoarse )
    filesCoarse.sort()

    # concatenate fine and coarse
    files = filesFine + filesCoarse
    # files = filesFine # use this line if fine is enough for now.

    # quit if we did not find anything
    if not files:
        printError( "Did not find any files. Please consult the README." )

    # a bit verbose
    print("Processing {} annotation files".format(len(files)))

    # iterate through files
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    for phase in ["train", "val"]:
        process_files = [f for f in files if phase in f]
        save_folder = os.path.join(save_root, phase)
        os.makedirs(save_folder, exist_ok=True)
        for level in range(0, 20):
            progress = 0
            save_folder_level = os.path.join(save_folder, f"L={level}", "ori")
            for f in process_files:
                if phase not in f:
                    continue
                # create the output filename
                dst = f.replace( "_polygons.json" , "_instanceBoundary.png" )
                file_name=os.path.basename(dst)
                dst = os.path.join(save_folder_level, file_name)
                # dst = dst.replace("gtFine", "boundaries")

                # if os.path.exists(dst):
                #     progress += 1
                #     print("\rProgress: {:>3} %".format( progress * 100 / len(process_files) ), end=' ')
                #     sys.stdout.flush()
                #     continue

                # do the conversion
                try:
                    json2boundary_prior( f , dst , "trainIds", level=level)
                    if resize:
                        boundary = cv2.imread(dst)
                        boundary = cv2.resize(boundary, size1)
                        boundary = cv2.resize(boundary, size2)
                        boundary = (boundary == 255).astype("uint8")
                        cv2.imwrite(dst, boundary * 255)
                except:
                    print("Failed to convert: {}".format(f))
                    raise

                # status
                progress += 1
                print(f"\rProgress phase: {phase} | level {level}: ({progress}/{len(process_files)}) {progress * 100 / len(process_files):.3f} %", end=' ')
                sys.stdout.flush()


# call the main
if __name__ == "__main__":
    main()

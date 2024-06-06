"""打印基于 BPG 压缩图像和语义分割图 SSM 的图像重建性能"""
import os
import sys

folder = sys.argv[1]
print("\t\tBPP\tBPP(SSM)\tBPP(Com)\tmIoU\tFID")
for step_folder in sorted(os.listdir(folder)):
	result_folder = os.path.join(folder, step_folder)
	result_path = os.path.join(result_folder, "performance.txt")
	with open(result_path, 'r') as f:
		lines = f.readlines()
	BPP, BPP_SSM, BPP_Com, mIoU, FID = 0, 0, 0, 0, 0
	for line in lines:
		if "BPP:" in line:
			BPP = float(line.strip().split(": ")[-1])
		if "BPP(SSM):" in line:
			BPP_SSM = float(line.strip().split(": ")[-1])
		if "BPP(COM):" in line:
			BPP_Com = float(line.strip().split(": ")[-1])
		if "mIoU:" in line:
			mIoU = float(line.strip().split(": ")[-1])
		if "FID:" in line:
			FID = float(line.strip().split(": ")[-1])
	print(f"{step_folder}:\t{BPP}\t{BPP_SSM}\t\t{BPP_Com}\t\t{mIoU}\t{FID}")

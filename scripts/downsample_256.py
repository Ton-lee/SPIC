import os
from PIL import Image
import tqdm

def resize_images(source_dir, target_dir, width=512, height=256):
    # 遍历源文件夹中的所有文件和子文件夹
    for root, _, files in os.walk(source_dir):
        for file in tqdm.tqdm(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # 构造源文件路径
                source_file_path = os.path.join(root, file)
                
                # 构造目标文件路径
                relative_path = os.path.relpath(root, source_dir)
                target_file_dir = os.path.join(target_dir, relative_path)
                target_file_path = os.path.join(target_file_dir, file)
                
                # 创建目标文件夹（如果不存在）
                os.makedirs(target_file_dir, exist_ok=True)
                
                # 打开、调整大小并保存图像
                try:
                    with Image.open(source_file_path) as img:
                        resized_img = img.resize((width, height), Image.BILINEAR)
                        resized_img.save(target_file_path)
                        print(f"Processed: {source_file_path} -> {target_file_path}")
                except Exception as e:
                    print(f"Error processing {source_file_path}: {e}")

if __name__ == "__main__":
    source_dir = "/home/Users/dqy/Dataset/Cityscapes/leftImg8bit_merged/"
    target_dir = "/home/Users/dqy/Dataset/Cityscapes/leftImg8bit_merged(512x256)/"
    
    # 执行图像调整
    resize_images(source_dir, target_dir)


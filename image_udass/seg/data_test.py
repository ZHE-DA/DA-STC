from PIL import Image, UnidentifiedImageError  
import os  
  
def find_unidentified_images(folder_path):  
    """  
    遍历指定文件夹及其所有子文件夹中的所有图片文件，并打印出无法识别的图片名称。  
  
    :param folder_path: 包含图片的文件夹路径  
    """  
    # 遍历指定文件夹及其所有子文件夹  
    for dirpath, dirnames, filenames in os.walk(folder_path):  
        for filename in filenames:  
            # 检查文件扩展名，只处理常见的图片格式  
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  
                file_path = os.path.join(dirpath, filename)  
                try:  
                    # 尝试打开图片  
                    with Image.open(file_path) as img:  
                        # 打印图片的尺寸（可选）  
                        # print(f"成功打开: {file_path}, 尺寸: {img.size}")  
                        pass
                except UnidentifiedImageError:  
                    # 捕获无法识别的图像错误  
                    print(f"无法识别的图片: {file_path}")  
  
# 指定包含图片的文件夹路径  
folder_path = '/home/customer/Desktop/ZZ/MIC-master/seg/data'  
find_unidentified_images(folder_path)
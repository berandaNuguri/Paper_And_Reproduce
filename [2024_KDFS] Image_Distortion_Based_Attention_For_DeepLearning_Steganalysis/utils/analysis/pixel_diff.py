from PIL import Image
import numpy as np
import rawpy
import matplotlib.pyplot as plt
import os

def calculate_pixel_difference(image_path1, image_path2):
    image_path1 = os.path.normpath(image_path1)
    data_format = os.path.basename(image_path1).split('.')[-1]
    
    image_path2 = os.path.normpath(image_path2)
    data_format2 = os.path.basename(image_path2).split('.')[-1]
    
    if data_format == 'dng' or data_format == 'tiff':
        with rawpy.imread(image_path1) as raw:
            rgb = raw.postprocess(output_color=rawpy.ColorSpace.sRGB,
                                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                                no_auto_bright=False,
                                use_camera_wb=True)
            img1 = np.array(rgb, dtype=np.uint8)
    else:
        img1 = Image.open(image_path1)

    
    if data_format2 == 'dng' or data_format2 == 'tiff':
        with rawpy.imread(image_path2) as raw:
            rgb = raw.postprocess(output_color=rawpy.ColorSpace.sRGB,
                                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                                no_auto_bright=False,
                                use_camera_wb=True)
            img2 = np.array(rgb, dtype=np.uint8)
    else:
        img2 = Image.open(image_path2)
    img1_array = np.array(img1, dtype=np.uint8)
    
    img2_array = np.array(img2, dtype=np.uint8)
    

    if img1_array.shape != img2_array.shape:
        raise ValueError("Images do not have the same dimensions")

    difference = np.abs(img1_array - img2_array)
    
    change_rate = np.sum(img1_array != img2_array) / img1_array.size
    
    print(f'Pixel Change Rate: {change_rate}%')
    
    img1_array = Image.fromarray(img1_array)
    img2_array = Image.fromarray(img2_array)
    difference_array = Image.fromarray(difference)
    
    
    img1_array.save('./utils/analysis/results/img1.png')
    img2_array.save('./utils/analysis/results/img2.png')
    difference_array.save('./utils/analysis/results/difference.png')
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img1)
    plt.title("Cover")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img2)
    plt.title("Stego")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(difference, cmap='gray')
    plt.title("Pixel Difference")
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    calculate_pixel_difference(r'C:\Users\AIM_DongSu\Downloads\Passlok\Cover_1.JPG',
                               r'C:\Users\AIM_DongSu\Downloads\Passlok\Stego_1.JPG')

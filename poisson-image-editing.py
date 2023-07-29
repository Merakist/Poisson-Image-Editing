import cv2
import os
import numpy as np
from glob import glob
from scipy.sparse import linalg as spl

IMG_EXTENSIONS = ["png", "jpeg", "jpg", "JPG", "gif", "tiff", "tif", "raw", "bmp"]
IN_FOLDER = "input"
OUT_FOLDER = "output"


def get_filename(prefix):
    filename = sum(map(glob, [prefix + ext for ext in IMG_EXTENSIONS]), [])
    return filename


def preview(source, target, mask):
    return (source * mask + target * (1 - mask))


def poisson_blending(source, target, mask):






if __name__ == '__main__':

    # Get image paths
    subfolders = os.walk(IN_FOLDER)

    for dirpath, dirnames, fnames in subfolders:
        image_dir = os.path.split(dirpath)[-1]
        output_dir = os.path.join(OUT_FOLDER, image_dir)
        print("Processing images in {}...".format(image_dir))

        source_names = get_filename(os.path.join(dirpath, "*source."))
        target_names = get_filename(os.path.join(dirpath, "*target."))
        mask_names = get_filename(os.path.join(dirpath, "*mask."))

        if not len(source_names) == len(target_names) == len(mask_names) == 1:
            print("Error: There should be exactly one source, target and mask image in each folder.")
            break

        # Read the source and target images and convert them to numpy arrays
        source_img = cv2.imread(source_names[0], cv2.IMREAD_COLOR)
        target_img = cv2.imread(target_names[0], cv2.IMREAD_COLOR)
        mask_img = cv2.imread(mask_names[0], cv2.IMREAD_GRAYSCALE)

        # Convert mask to binary image
        mask = np.atleast_3d(mask_img).astype(np.float32) / 255.0
        mask[mask != 1] = 0
        mask = mask[:, :, 0]

        # Get the number of color channel(s) for poisson operation
        channels = source_img.shape[-1]

        result_stack = [preview(source_img[:, :, i], target_img[:, :, i], mask) for i in range(channels)]
        composite_img = cv2.merge(result_stack)
        cv2.imwrite("./output/composite.jpg", composite_img)

    print("Poisson blending completed.")

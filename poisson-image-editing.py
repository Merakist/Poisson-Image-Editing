import cv2
import os
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spl
from glob import glob

IMG_EXTENSIONS = ["png", "jpeg", "jpg", "JPG", "gif", "tiff", "tif", "raw", "bmp"]
IN_FOLDER = "input"
OUT_FOLDER = "output"


def get_filename(prefix):
    filename = sum(map(glob, [prefix + ext for ext in IMG_EXTENSIONS]), [])
    return filename


def preview(source, target, mask):
    return (source * mask + target * (1 - mask))


def bool_pixel_on_boundary(index, mask):
    x, y = index
    if mask[x, y] == 1:
        for pixel in get_neighbors(index):
            if mask[pixel] == 0:
                return True
    return False


def get_neighbors(index):
    x, y = index
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    return neighbors


def laplacian(src, index):
    i, j = index
    value = 4 * src[i, j] - src[i - 1, j] \
        - src[i + 1, j] - src[i, j - 1] - src[i, j + 1]
    return value


def poisson_sparse_matrix(indicies):
    # N = Number of points in mask
    N = len(indicies)
    A = sparse.lil_matrix((N, N))

    for i, index in enumerate(indicies):
        # Diagonal (Points on boundary)
        A[i, i] = 4
        # Off-diagonal
        for coord in get_neighbors(index):
            if coord not in indicies:
                continue
            j = indicies.index(coord)
            A[i, j] = -1
    return A


def poisson_blending(source, target, mask):
    # Get mask indices
    indicies = list(zip(np.nonzero(mask)[0], np.nonzero(mask)[1]))

    # Create poisson sparse matrix A
    N = len(indicies)
    A = poisson_sparse_matrix(indicies)

    # Create result matrix b
    b = np.zeros(N)
    for i, index in enumerate(indicies):
        # Pixels inside Omega region
        # TODO: Compare the gradient of source and target on index to choose laplacian src
        b[i] = laplacian(source, index)
        # Pixels on the boundary
        if mask[index] == 1 and bool_pixel_on_boundary(index, mask):
            for pixel in get_neighbors(index):
                if mask[pixel] == 0:
                    b[i] += target[pixel]

    # Solve Ax = b
    x = spl.cg(A, b)

    # Copy target image
    composite = np.copy(target).astype(int)
    # Place source area onto target image
    for i, index in enumerate(indicies):
        composite[index] = x[0][i]
    return composite


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

        # result_stack = [preview(source_img[:, :, i], target_img[:, :, i], mask) for i in range(channels)]
        result_stack = [poisson_blending(source_img[:, :, i], target_img[:, :, i], mask) for i in range(channels)]
        composite_img = cv2.merge(result_stack)
        cv2.imwrite("./output/composite.jpg", composite_img)

    print("Poisson blending completed.")

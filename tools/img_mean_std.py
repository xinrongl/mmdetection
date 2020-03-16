import argparse
from pathlib import Path
from PIL import Image
import numpy as np
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    pass
	
	
def args_parse():
    parser = argparse.ArgumentParser("Calculate image mean and std.")
    parser.add_argument("--images_dir", help="Directory contains a list of images")
    _args = parser.parse_args()
    return _args


def images_stats(imgs):
    mean = []
    std = []
    for img in tqdm(imgs):
        img_np = np.array(Image.open(img))/255
        rgb_mean = np.mean(img_np, axis=(0, 1))
        rgb_std = np.std(img_np, axis=(0, 1))

        mean.append(rgb_mean)
        std.append(rgb_std)
    mean = np.array(mean)
    std = np.array(std)
    return mean, std
	
	
if __name__ == "__main__":
    imgs = list(sorted(Path(args.images_dir).rglob("*.jpg")))
    print("process %s images:" % len(imgs))
    mean, std = images_stats(imgs)
    u = np.mean(mean, axis=0)
    sigma = np.mean(std, axis=0)
    print("target mean: ", np.around(u, decimals=3))
    print("target std:  ", np.around(sigma, decimals=3))

import random
import shutil
from pathlib import Path
import argparse


def _path(_dir):
    path_obj = Path(_dir)

    if not path_obj.exists():
        path_obj.mkdir(parents=True, exist_ok=True)
    if path_obj.is_dir():
        return path_obj
    else:
        raise argparse.ArgumentTypeError("Not a directory")


def args_parser():
    parser = argparse.ArgumentParser("split training data to train and validate")
    parser.add_argument("-d", "--data_dir", help="folder contain train images and training labels", type=_path)
    parser.add_argument("-s", "--seed", help="random seed for train val split", default=1, type=float)
    parser.add_argument("-r", "--ratio", help="validate ratio for train val split", default=0.05, type=float)
    parser.add_argument("-o", "--out_dir", help="output directory for train and validate data", type=_path)
    _args, _ = parser.parse_known_args()
    return _args


def _mk_dir(out_path):
    """
    Make training and validate directory to store data in PASCOL format
    :param out_path: _path obj
    :return: list contain train_img, train_label, val_img, val_label
    """
    train_img = out_path.joinpath("train", "image")
    train_label = out_path.joinpath("train", "label")
    val_img = out_path.joinpath("validate", "image")
    val_label = out_path.joinpath("validate", "label")

    _dirs = [train_img, train_label, val_img, val_label]
    for d in _dirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=False)
    return _dirs


def train_val_split(data_path):
    """
    Main function to split training data
    :param data_path: str
    :return: train dict and val dict containing images and labels file name
    """
    data_path = Path(data_path)
    imgs = sorted(list(data_path.rglob("*.jpg")))
    labels = sorted(list(data_path.rglob("*.xml")))
    print("Images: %s" % len(imgs))
    assert len(imgs) == len(labels)

    data = dict(zip(imgs, labels))
    keys = list(data.keys())
    random.shuffle(keys)
    data_shuffled = {}
    data_shuffled.update({k: data[k] for k in keys})

    train_dic = {}
    val_dic = {}
    threshold = int(len(imgs) * (1 - args.ratio))
    for i, (k, v) in enumerate(data_shuffled.items()):
        if i <= threshold:
            train_dic.update({k: v})
        else:
            val_dic.update({k: v})

    return train_dic, val_dic


def main():
    train_dic, val_dic = train_val_split(data_path=args.data_dir)
    dirs = _mk_dir(out_path=args.out_dir)

    for im, l in train_dic.items():
        shutil.copy(im, dirs[0].joinpath(im.parts[-1]))
        shutil.copy(l, dirs[1].joinpath(l.parts[-1]))
    for im, l in val_dic.items():
        shutil.copy(im, dirs[2].joinpath(im.parts[-1]))
        shutil.copy(l, dirs[3].joinpath(l.parts[-1]))

    print("Success: %s:" % args.out_dir)
    print("Train %s:" % len(train_dic))
    print("Validate %s:" % len(val_dic))
    return None


if __name__ == "__main__":
    args = args_parser()
    random.seed(args.seed)
    main()

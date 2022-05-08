import shutil
import random
from pathlib import Path

raw_dirs = ["bounding_box_train", "bounding_box_test", "query", "gt_bbox"]
data_path = Path("../data")
raw_path = data_path / "raw"

raw_dirs = [raw_path / _dir for _dir in raw_dirs]

train_dir = data_path / "train"
val_dir = data_path / "val"

if not train_dir.exists():
    train_dir.mkdir()

if not val_dir.exists():
    val_dir.mkdir()

num_files = {}

for dir in raw_dirs:
    dir_list = list(dir.glob("*.jpg"))
    random.shuffle(dir_list)

    for file in dir_list:
        cat = file.name.split("_")[0]
        if cat == "-1" or cat == "0000":
            continue
        cat = int(cat)
        if cat not in num_files:
            num_files[cat] = 0
        if random.randint(0, 1) < 0.3 and num_files[cat] < 5:
            shutil.copy(file, val_dir / (file.name))
            num_files[cat] += 1
        else:
            shutil.copy(file, train_dir / (file.name))

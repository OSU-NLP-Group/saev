import os.path

import beartype
import torch
from jaxtyping import Bool, jaxtyped
from torch import Tensor

import saev.data


@jaxtyped(typechecker=beartype.beartype)
def load_attrs(root: str, *, is_train: bool) -> Bool[Tensor, "N T"]:
    wanted_split = "train" if is_train else "test"
    image_folder_dataset = saev.data.images.ImageFolderDataset(
        os.path.join(root, wanted_split)
    )

    img_path_to_img_id = {}
    with open(os.path.join(root, "metadata", "images.txt")) as fd:
        for line in fd:
            img_id, img_path = line.split()
            img_path_to_img_id[img_path] = int(img_id)

    img_id_in_split = set()
    with open(os.path.join(root, "metadata", "train_test_split.txt")) as fd:
        for line in fd:
            img_id, is_train_str = line.split()
            if is_train and is_train_str == "0":
                continue
            if not is_train and is_train_str == "1":
                continue

            img_id_in_split.add(int(img_id))

    img_id_to_i = {}
    for i, (path, _) in enumerate(image_folder_dataset.samples):
        path, filename = os.path.split(path)
        path, cls = os.path.split(path)
        img_id = img_path_to_img_id[os.path.join(cls, filename)]
        img_id_to_i[img_id] = i

    attr_id_to_attr_name = {}
    with open(os.path.join(root, "metadata", "attributes", "attributes.txt")) as fd:
        for line in fd:
            attr_id, attr_name = line.split()
            attr_id_to_attr_name[int(attr_id)] = attr_name

    attr_id_to_i = {
        attr_id: i for i, attr_id in enumerate(sorted(attr_id_to_attr_name))
    }

    y_true_NT = torch.empty(
        (len(img_id_in_split), len(attr_id_to_attr_name)), dtype=bool
    )

    # From certainties.txt
    # 1 not visible
    # 2 guessing
    # 3 probably
    # 4 definitely
    certainty_keep = {3, 4}

    fpath = os.path.join(root, "metadata", "attributes", "image_attribute_labels.txt")
    with open(fpath) as fd:
        for line in fd:
            # Explanation of *_: Sometimes there's an extra field (worker_id) but not on every line. So *_ collects all extra fields besides the 5 that are manually unpacked, then we ignore it.
            img_id, attr_id, present, certainty_id, *_, time_s = line.split()
            img_id, attr_id, certainty_id = int(img_id), int(attr_id), int(certainty_id)

            if img_id not in img_id_in_split:
                continue

            i = img_id_to_i[img_id]
            j = attr_id_to_i[attr_id]
            y_true_NT[i, j] = certainty_id in certainty_keep

    return y_true_NT

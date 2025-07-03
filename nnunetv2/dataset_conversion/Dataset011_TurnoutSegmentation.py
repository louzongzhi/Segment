import multiprocessing
import shutil
import numpy as np
from multiprocessing import Pool
from PIL import Image

from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io


def load_and_convert_case(input_image: str, input_seg: str, output_image: str, output_seg: str,
                          min_component_size: int = 50):
    seg = np.array(Image.open(input_seg).convert('L'))
    seg[seg == 15] = 1
    seg[seg == 38] = 2
    seg[seg == 75] = 3
    seg[seg == 113] = 4
    io.imsave(output_seg, seg, check_contrast=False)
    shutil.copy(input_image, output_image)


if __name__ == "__main__":
    # extracted archive from https://www.kaggle.com/datasets/insaff/massachusetts-roads-dataset?resource=download
    source = '/home/dell/桌面/seg/nnUNet_raw/Task011_TurnoutSegmentation'

    dataset_name = 'Dataset011_TurnoutSegmentation'

    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    imagests = join(nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    labelsts = join(nnUNet_raw, dataset_name, 'labelsTs')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_source = join(source, 'training')
    test_source = join(source, 'testing')

    with multiprocessing.get_context("spawn").Pool(8) as p:

        # not all training images have a segmentation
        valid_ids = subfiles(join(train_source, 'output'), join=False, suffix='png')
        num_train = len(valid_ids)
        r = []
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         join(train_source, 'input', v),
                         join(train_source, 'output', v),
                         join(imagestr, v[:-4] + '_0000.png'),
                         join(labelstr, v),
                         50
                     ),)
                )
            )

        # test set
        valid_ids = subfiles(join(test_source, 'output'), join=False, suffix='png')
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         join(test_source, 'input', v),
                         join(test_source, 'output', v),
                         join(imagests, v[:-4] + '_0000.png'),
                         join(labelsts, v),
                         50
                     ),)
                )
            )
        _ = [i.get() for i in r]

    generate_dataset_json(
        join(nnUNet_raw, dataset_name),
        {0: 'R', 1: 'G', 2: 'B'},
        {
            'background': 0,
            'Point_Machine': 1,
            'Switch_Rail': 2,
            'Stock_Rail': 3,
            'Tie_Rod': 4
        },
        num_train,
        '.png',
        dataset_name=dataset_name
    )

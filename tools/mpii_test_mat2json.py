import argparse
import json
import time

from scipy.io import loadmat


def parse_args():
    parser = argparse.ArgumentParser(
        description='Converting the mpii ground truth .mat file to .json file.')
    parser.add_argument('gt_mat_file',
                        default="data/mpii/annolist_dataset_v12.mat",
                        help='input prediction mat file.')
    parser.add_argument(
        'gt_json_file_wo_anno',
        default="data/mpii/annotations/test.json",
        help='input ground-truth json file to get the image name. '
        'Default: "data/mpii/mpii_val.json" ')
    parser.add_argument('output_json_file_with_anno',
                        default="data/mpii/test_with_anno.json",
                        help='output converted json file.')
    args = parser.parse_args()
    return args


def save_json(list_file, path):
    with open(path, 'w') as f:
        json.dump(list_file, f, indent=4)
    return 0


def convert_mat(gt_mat_file, gt_json_file_wo_anno, output_json_file):
    res = loadmat(gt_mat_file)
    import pdb
    pdb.set_trace()
    preds = res['preds']
    N = preds.shape[0]

    with open(gt_json_file_wo_anno) as anno_file:
        anno = json.load(anno_file)

    assert len(anno) == N

    instance = {}

    for pred, ann in zip(preds, anno):
        ann.pop('joints_vis')
        ann['joints'] = pred.tolist()

    instance['annotations'] = anno
    instance['info'] = {}
    instance['info']['description'] = 'Converted MPII prediction.'
    instance['info']['year'] = time.strftime('%Y', time.localtime())
    instance['info']['date_created'] = time.strftime('%Y/%m/%d',
                                                     time.localtime())

    save_json(instance, output_json_file)


def main():
    args = parse_args()
    convert_mat(args.gt_mat_file, args.gt_json_file_wo_anno, args.output_json_file_with_anno)


if __name__ == '__main__':
    main()

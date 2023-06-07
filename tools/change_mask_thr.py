import os
import json
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--thr', type=float, default=0)

    args = parser.parse_args()

    output_dir = args.output_dir
    thr = args.thr

    org_json = os.path.join(output_dir, 'msrcnn_pseudo_label.json')
    filename = os.path.join(output_dir, 'msrcnn_pseudo_label_{}.json'.format(thr))
    res = {}
    with open(org_json, 'r', encoding='utf-8') as f:
        a = json.load(f)

    print("org len: " + str(len(a['annotations'])))

    res['annotations'] = []
    res['categories'] = a['categories']
    res['images'] = a['images']

    cls_id = {}
    j = 1
    for i in tqdm(range(len(a['annotations'])),total=len(a['annotations'])):
        if a['annotations'][i]['score'] < thr:
            pass

        else:
            a['annotations'][i]['id'] = j
            j += 1
            res['annotations'].append(a['annotations'][i])

    print("after thr len: " + str(len(res['annotations'])))
    with open(filename,'w') as file_obj:
        json.dump(res, file_obj)

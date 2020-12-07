import os
import base64
import numpy as np
import csv
import sys
import argparse

csv.field_size_limit(sys.maxint)
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

def main(args):
    count = 0
    with open(args.infeats, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            if count % 1000 == 0:
                print(count)
            count += 1

            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            HAVE_ERROR=False
            for field in ['boxes', 'features']:
                missing_padding = 4 - len(item[field]) % 4
                if missing_padding:
                    item[field] += b'=' * missing_padding
                try:
                    item[field] = np.frombuffer(base64.decodestring(item[field]),dtype=np.float32).reshape((item['num_boxes'],-1))
                except Exception as e:
                    print(e)
                    print('%d has wrong field'%(int(item['image_id'])))
                    HAVE_ERROR=True
            image_id = item['image_id']
            
            feats = item['features']
            if not HAVE_ERROR:
                np.savez_compressed(os.path.join(args.outfolder, str(image_id)), feat=feats)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--infeats', default='../bottom_up_data/test2014/test2014_resnet101_faster_rcnn_genome.tsv.2', help='image features')
    parser.add_argument('--outfolder', default='../mscoco/feature/up_down_10_100/test2014/', help='output folder')

    args = parser.parse_args()
    main(args)

'''
test2014_resnet101_faster_rcnn_genome.tsv.0: 321486 has wrong field
test2014_resnet101_faster_rcnn_genome.tsv.1: 300104 has wrong field
test2014_resnet101_faster_rcnn_genome.tsv.2: 147295 has wrong field
'''
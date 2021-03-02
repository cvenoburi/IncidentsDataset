import os
import pprint
import glob
import json
from tqdm import tqdm
import cv2
import torch
#import matplotlib.pyplot as plt
#%matplotlib inline

from architectures import (
    FilenameDataset,
    get_incidents_model,
    update_incidents_model_with_checkpoint,
    update_incidents_model_to_eval_mode,
    get_predictions_from_model
)
from parser import get_parser, get_postprocessed_args

from utils import get_index_to_incident_mapping, get_index_to_place_mapping

# model
CONFIG_FILENAME = "configs/eccv_final_model"
CHECKPOINT_PATH_FOLDER = "pretrained_weights/"

# call command
# python detect_incidents.py --config=configs/eccv_final_model --checkpoint_path=pretrained_weights/ --mode=test --num_gpus=1 --topk=5 --images_file=example_images_file.txt --images_path=example_images/ --output_file=example_images_incidents.tsv
parser = get_parser()
args = parser.parse_args()
args = get_postprocessed_args(args)

# data
with open(args.images_file,"r") as f:
    image_filenames = [l.strip() for l in f.readlines() if l.strip()]

incidents_model = get_incidents_model(args)
update_incidents_model_with_checkpoint(incidents_model, args)
update_incidents_model_to_eval_mode(incidents_model)

# Set up the data loader for quickly loading images to run inference with.
print("num images: {}".format(len(image_filenames)))
targets = [image_filenames[i] for i in range(len(image_filenames))]
dataset = FilenameDataset(image_filenames, targets, args.images_path)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4
)

inference_dict = {}
for idx, (batch_input, image_paths) in tqdm(enumerate(loader)):
    # run the model, get the output, set the inference_dict
    output = get_predictions_from_model(
        args,
        incidents_model,
        batch_input,
        image_paths,
        get_index_to_incident_mapping(),
        get_index_to_place_mapping(),
        inference_dict,
        topk=args.topk
    )

# todo: inference_dict contains numpy arrays which are not JSON serializable,
# either convert them to lists before saving, or change the output format to something else, e.g., tsv
with open(args.output_file, "w") as write_file:
    for image_filename in inference_dict:
        out_line = image_filename
        for i in range(args.topk):
            out_line += "\t{}\t{}".format(inference_dict[image_filename]['incidents'][i],
                                          inference_dict[image_filename]['incident_probs'][i])
        write_file.write("{}\n".format(out_line))
print("output saved into \'{}\'".format(args.output_file))



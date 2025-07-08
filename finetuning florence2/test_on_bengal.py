# IMPORTS

import torch
import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
from maestro.trainer.models.florence_2.core import train
from maestro.trainer.models.florence_2.inference import predict
from maestro.trainer.models.florence_2.checkpoints import OptimizationStrategy, load_model
import supervision as sv
from supervision.geometry.core import Position
from glob import glob
import shutil
from IPython.display import Image
from maestro.trainer.models.florence_2.checkpoints import OptimizationStrategy, load_model
import matplotlib.pyplot as plt
from maestro.trainer.common.datasets.coco import COCODataset
from maestro.trainer.models.florence_2.loaders import evaluation_collate_fn, train_collate_fn
from maestro.trainer.models.florence_2.detection import (
    detections_to_prefix_formatter,
    detections_to_suffix_formatter,
)
from functools import partial
from maestro.trainer.common.datasets.core import create_data_loaders
from maestro.trainer.models.florence_2.inference import predict_with_inputs
from tqdm import tqdm
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def add_bboxes(label_path, image_id, annotation_id):
    bbox_dict = []

    image_size = 320

    bboxes = np.loadtxt(label_path, ndmin = 2)
    bboxes = bboxes * image_size
    # print(bboxes)

    for bbox in bboxes:
        xmin = min(bbox[[1,3,5,7]])
        ymin = min(bbox[[2,4,6,8]])
        xmax = max(bbox[[1,3,5,7]])
        ymax = max(bbox[[2,4,6,8]])
        # print(xmin, ymin, xmax, ymax)
        width = int(xmax - xmin)
        height = int(ymax - ymin)
        bbox_dict.append({"id": annotation_id,
        "image_id": image_id,
        "category_id": 0,
        "bbox": [
            int(xmin),
            int(ymin),
            width,
            height
            ],
        "area": width * height,
        "segmentation": [],
        "iscrowd": 0
        })
        annotation_id += 1
    
    return bbox_dict, annotation_id

# CREATING ANNOTATIONS

def create_training_annotations(num_nbg, num_bg):
    type = "train"
    directory = f'//{type}'
    images_path = sorted(glob(f'//{type}/*'))
    labels_path = f'{directory}/labels'
    image_size = 320
    coco_dict = {}

    coco_dict['info'] = {
        "year": "2024",
        "version": "0",
        "description": f"coco format of yolo labels of {type} set",
        "contributor": "",
        "url": "",
        "date_created": ""
    }

    coco_dict['licenses'] = [{
        "id": 1,
        "url": "",
        "name": ""
        }]

    coco_dict['categories'] = [
        {
        "id": 0,
        "name": "brick kilns with chimney",
        "supercategory": "none"
        },
        {
        "id": 1,
        "name": "background",
        "supercategory": "none"
        },
    ]

    coco_dict['images'] = []

    coco_dict['annotations'] = []

    annotation_id = 0
    for image_id, image_path in enumerate(images_path):
        image_name = os.path.basename(image_path)
        if image_name[-3:] == 'tif' or image_name[-3:] == 'png':
            coco_dict['images'].append({
                "id": image_id,
                "license": 1,
                "file_name": image_name,
                "height": image_size,
                "width": image_size,
                "date_captured": ""
            })

        label_path = os.path.join(labels_path, image_name[:-4]+'.txt')
        if os.path.exists(label_path):
            bbox_dict, annotation_id = add_bboxes(label_path, image_id, annotation_id)
            coco_dict['annotations'] += bbox_dict
        else:
            coco_dict['annotations'].append({"id": annotation_id,
                                        "image_id": image_id,
                                        "category_id": 1,
                                        "bbox": [0, 0, image_size, image_size],
                                        "area": image_size * image_size,
                                        "segmentation": [],
                                        "iscrowd": 0
                                        })
            annotation_id += 1
    fp = open('/', 'w')
    json.dump(coco_dict, fp)
    fp.close()

def add_class_ids_and_confidence(detection):
    CLASSES = ['brick kilns with chimney', 'background']
    detection.class_id = []
    detection.confidence = []
    for class_name in detection.data['class_name']:
        try:
            detection.class_id.append(CLASSES.index(class_name))
        except:
            detection.class_id.append(len(CLASSES)) # if class name is different from ['brick kilns with chimney', 'background'] assign class id other than 0 & 1
        detection.confidence.append(1) # florence_2 does not give confidence score so no use of conf_threshold
        
    return detection


for id in [42, 55]:
    print(id)
    print('testing on west bengal dataset')
    processor, model = load_model(
        model_id_or_path=f"./training/florence_2/{id}/checkpoints/latest",
        revision = "refs/heads/main"
        )
    test_dataset = COCODataset(
        annotations_path="./dynamic_lucknow_coco_train_test/train/_annotations.coco.json",
        images_directory_path="./dynamic_lucknow_coco_train_test/train",
    )

    CLASSES = test_dataset.classes
    print(CLASSES)
    train_loader, valid_loader, test_loader = create_data_loaders(
                    dataset_location= './west_bengal_test_set',
                    train_batch_size= 32,
                    train_collect_fn= partial(train_collate_fn, processor=processor),
                    # train_num_workers=10,
                    test_batch_size= 4,
                    test_collect_fn= partial(evaluation_collate_fn, processor=processor),
                    detections_to_prefix_formatter=detections_to_prefix_formatter,
                    detections_to_suffix_formatter=detections_to_suffix_formatter,
                    )
    predictions = []
    targets = []

    for input_ids, pixel_values, images, prefixes, suffixes in tqdm(test_loader):
        generated_texts = predict_with_inputs(model=model, processor=processor, input_ids=input_ids, pixel_values=pixel_values, device = model.device)

        for generated_text in generated_texts:
            predicted_result = processor.post_process_generation(text=generated_text, task="<OD>", image_size=(images[0].width, images[0].height))
            predicted_result = sv.Detections.from_vlm(vlm='florence_2',
                    result=predicted_result,
                    resolution_wh=(320, 320))
            
            if len(predicted_result.xyxy) == 0: # if florence2 gives no detection then consider it as background
                predicted_result.xyxy = np.array([[0,0, images[0].width, images[0].height]])
                predicted_result.confidence = [1]
                predicted_result.data = {'class_name': ['background']}

            predicted_result = add_class_ids_and_confidence(predicted_result)
            predictions.append(predicted_result)

        for suffix in suffixes:
            target_result = processor.post_process_generation(text=suffix, task="<OD>", image_size=(images[0].width, images[0].height))
            target_result = sv.Detections.from_vlm(vlm='florence_2',
                    result=target_result,
                    resolution_wh=(320, 320))
            target_result = add_class_ids_and_confidence(target_result)
            targets.append(target_result)

    df = pd.DataFrame({}, columns = ['IoU', 'Precision', 'Recall', 'F1 score', 'TP', 'FP', 'FN', 'Kiln instances'])
    for iou in [0.1,0.3,0.5,0.7]:
        confusion_matrix = sv.ConfusionMatrix.from_detections(
            predictions=predictions,
            targets=targets,
            classes=CLASSES,
            conf_threshold = 0.25, # florence_2 does not give confidence score so no use of conf_threshold  
            iou_threshold=iou
        )

        # calculate precision recall and f1-score
        cm = confusion_matrix.matrix
        tp = cm[0][0]
        predicted_positives = cm[:,0].sum()
        actual_positives = cm[0, :].sum()
        precision = tp/ (predicted_positives + 1e-9)
        recall = tp/ (actual_positives + 1e-9)
        f1_score = 2*precision*recall / (precision + recall + 1e-9)
        false_positives = predicted_positives - tp

        df = pd.concat([df, pd.DataFrame({'IoU': iou, 'Precision': precision, 'Recall': recall, 'F1 score': f1_score, 'TP': tp, 'FP': false_positives, 'FN': actual_positives - tp, 'Kiln instances': actual_positives}, index = [0])])
    print(df)

    print(f'\n\nPlot of Confusion matrix at IoU {iou}')
    _ = confusion_matrix.plot()
    

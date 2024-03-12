import argparse
import os
import json
import numpy as np
from utils.predictors import VisualizationDemo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import _create_text_labels
from tqdm import tqdm

eva02_L_lvis_sys_o365_config_path = ("projects/ViTDet/configs/eva2_o365_to_lvis/"
                                     "eva2_o365_to_lvis_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py")
eva02_L_lvis_sys_o365_ckpt_path = "all_checkpoints/eva02_L_lvis_sys_o365.pth"

all_configs = [eva02_L_lvis_sys_o365_config_path]
all_opts = [f"train.init_checkpoint={eva02_L_lvis_sys_o365_ckpt_path}"]


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--key_frame_dir_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--opts", required=False, default="")

    args = parser.parse_args()

    return args


def json_serializable(data):
    if isinstance(data, np.float32):  # if it's a np.float32
        return round(float(data), 2)  # convert to python float
    elif isinstance(data, np.ndarray):  # if it's a np.ndarray
        return data.tolist()  # convert to python list
    else:  # for other types, let it handle normally
        return data


def setup(config_file, opts):
    cfg = LazyConfig.load(config_file)
    cfg = LazyConfig.apply_overrides(cfg, [opts])
    return cfg


def run_inference(config_path, opts, key_frame_dir_path, confidence_threshold, model_name, output_dir_path):
    os.makedirs(f"{output_dir_path}/{model_name}", exist_ok=True)
    # Initiate cfg
    cfg = setup(config_path, opts)
    # Initiate the model (create, load checkpoints, put in eval mode)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.train.init_checkpoint)
    model.eval()
    # Run inference
    predictor = VisualizationDemo(model=model, min_size_test=800, max_size_test=1333, img_format="RGB",
                                  metadata_dataset="lvis_v1_train")

    for video_keyframe_dir in tqdm(os.listdir(key_frame_dir_path)):
        video_tags_dict = {}
        video_name = video_keyframe_dir
        video_tags_dict[video_name] = {}
        video_keyframe_dir_path = os.path.join(key_frame_dir_path, video_keyframe_dir)
        all_data = {}
        for i, frame in enumerate(os.listdir(video_keyframe_dir_path)):
            all_data[frame] = {}
            image_path = f"{video_keyframe_dir_path}/{frame}"

            image = read_image(image_path, format="RGB")
            predictions = predictor.run_on_image(image, confidence_threshold)

            bboxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
            scores = predictions.scores.numpy().tolist() if predictions.has("scores") else None
            labels = _create_text_labels(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None,
                                         None, predictor.metadata.get("thing_classes", None))

            final_bboxes = []
            for box in bboxes:
                box = [round(float(b), 2) for b in box]
                final_bboxes.append(box)

            all_image_predictions = []
            for j, box in enumerate(final_bboxes):
                prediction = {}
                prediction['bbox'] = box
                prediction['score'] = round(scores[j], 2)
                prediction['label'] = labels[j]
                all_image_predictions.append(prediction)

            # all_image_predictions = compute_depth(all_image_predictions, image_name[:-4])

            all_data[frame] = [{k: json_serializable(v) for k, v in prediction.items()} for prediction in
                               all_image_predictions]

        # Write all_data to a JSON file (file_wise)
        with open(f"{output_dir_path}/{model_name}/{video_name}.json", 'w') as f:
            json.dump(all_data, f)


def main():
    args = parse_args()
    key_frame_dir_path = args.key_frame_dir_path
    output_dir_path = args.output_dir_path

    os.makedirs(output_dir_path, exist_ok=True)

    run_inference(all_configs[0], all_opts[0], key_frame_dir_path, 0.6, "eva-02", output_dir_path)


if __name__ == "__main__":
    main()

import pickle 
import torch 
from torch.utils.data import Dataset
import numpy as np
import os
import ast
import csv
import natsort
from PIL import Image
from torchvision import transforms
import random
import json
from transformers import VideoMAEImageProcessor
import cv2 
from .constants_video import TRAIN_OBJECTS, VAL_OBJECTS, TEST_OBJECTS, HARDNESS_MAP, PROTRUSION_MAP, ELASTICITY_MAP, FRICTION_MAP, RANKS


def get_frames_videomae(video_path, video_processor, max_length=16, return_indices=False):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames <= 0:
        cap.release()
        raise ValueError(f"Video file has no frames or cannot read frame count: {video_path}")

    indices = np.linspace(0, num_frames - 1, num=max_length, dtype=int)
    unique_indices = sorted(list(set(indices)))

    video_frames = []
    sampled_indices_actual = [] 

    current_frame_idx = 0
    target_indices_iter = iter(unique_indices)
    target_idx = next(target_indices_iter, None)

    while True:
        ret, frame = cap.read()
        if not ret or target_idx is None:
            break


        if current_frame_idx == target_idx:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            video_frames.append(pil_image)
            sampled_indices_actual.append(current_frame_idx)
            target_idx = next(target_indices_iter, None)

        current_frame_idx += 1

    cap.release() # Release video capture object

    num_read_frames = len(video_frames)
    if num_read_frames == 0:
        raise ValueError(f"Cannot read any frames from video: {video_path}. Please check the file.")

    # Repeat last frame to reach max_length
    if num_read_frames < max_length:
        print(f"  Insufficient frames, copying last frame ({num_read_frames} < {max_length})")
        last_frame = video_frames[-1]
        last_index = sampled_indices_actual[-1]
        for _ in range(max_length - num_read_frames):
            video_frames.append(last_frame)
            sampled_indices_actual.append(last_index) 


    processed_output = video_processor(video_frames, return_tensors="pt")
    pixel_values = processed_output.pixel_values # (1, num_frames, C, H, W)
    pixel_values = pixel_values.squeeze(0) # (num_frames, C, H, W)


    if return_indices:
        return pixel_values, sampled_indices_actual
    return pixel_values


class CLIPPropertyUniqueDataset(Dataset):
    def __init__(self, video_processor, data_path, split_name, flip_p=0):
        super().__init__()
        print(f"Initializing CLIPPropertyUniqueDataset ({split_name})...")
        self.split_name = split_name
        self.flip_p = flip_p
        self.video_processor = video_processor
        self.properties = ["hardness", "protrusion", "elasticity", "friction"]
        json_path = [os.path.join(data_path, f"{self.split_name}_samples.json")]
        print(f"  Loading sample file: {json_path[0]}")
        for i in range(len(json_path)):
            if i == 0:
                with open(json_path[i]) as json_file:
                    self.samples = json.load(json_file)
                    json_file.close()
            else:
                with open(json_path[i]) as json_file:
                    samples_temp = json.load(json_file)
                    json_file.close()
                for k, v in samples_temp.items():
                    if k in self.samples.keys():
                        self.samples[k] += v
                    else:
                        self.samples[k] = v

        self.objects = []
        self.all_samples = []
        for k in self.samples.keys():
            if k not in TRAIN_OBJECTS + VAL_OBJECTS + TEST_OBJECTS:
                continue
            for v in self.samples[k]:
                self.objects.append(k)
                self.all_samples.append(v)
        print(f"  Loading complete, total {len(self.objects)} samples")
        

        if split_name == "train":
            indices = list(range(len(self.objects)))
            random.shuffle(indices)
            self.objects = [self.objects[i] for i in indices]
            self.all_samples = [self.all_samples[i] for i in indices]


    def get_frames_and_label(self, index):
        objects = self.objects[index]
        video_path = self.all_samples[index]
        try:
            pixel_values, indices = get_frames_videomae(video_path, self.video_processor, max_length=16, return_indices=True)
            objects_tactile_pixel_values = [pixel_values]
            all_indices = [indices]
            hardness_label = RANKS["hardness"][objects]
            protrusion_label = RANKS["protrusion"][objects]
            elasticity_label = RANKS["elasticity"][objects]
            friction_label = RANKS["friction"][objects]
            return objects_tactile_pixel_values, hardness_label, protrusion_label, elasticity_label, friction_label, all_indices
        except Exception as e:
            print(f"Error processing sample {index}: {e}")
            dummy_pixel_values = torch.zeros((16, 3, 224, 224))
            dummy_indices = [0] * 16
            return [dummy_pixel_values], 0, 0, 0, 0, [dummy_indices]
    
    def __len__(self): 
        return len(self.objects)

    def __getitem__(self, index):
        objects_tactile_pixel_values, hardness_label, protrusion_label, elasticity_label, friction_label, all_indices = self.get_frames_and_label(index)
        return objects_tactile_pixel_values, hardness_label, protrusion_label, elasticity_label, friction_label, all_indices


class TactileLLMDataset(Dataset):
    def __init__(self, video_processor, files, split_name, tokenizer):
        super().__init__()
        self.split_name = split_name
        self.tokenizer = tokenizer
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.pad_token = tokenizer.pad_token
        self.eos_token_number = self.tokenizer.encode(self.eos_token)
        self.video_processor = video_processor
        self.samples = None
        all_files = []
        for file in files:
            with open(file, 'r') as f:
                all_files += json.load(f)
        self.samples = all_files

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        question_type = sample[0]["question_type"]
        question_step = sample[0]["question_steps"]
        question = []
        tactile_paths = []
        for s in sample[1:-1]:
            if s["role"] == "ASSISTANT":
                question += [s["role"]] + [": "] + s["content"] + [f"{self.eos_token}\n"]
            else:
                question += [s["role"]] + [": "] + s["content"] + ["\n"]
            tactile_paths += s["video"]
        question += ["ASSISTANT: "]
        answer = "".join(sample[-1]["content"])
        answer_tokens = torch.tensor(self.tokenizer.encode(answer + f'{self.eos_token}'), dtype=torch.int64)[1:]

        all_tactile_pixel_values = []
        all_indices = []
        for t_path in tactile_paths:
            pixel_values, indices = get_frames_videomae(t_path, self.video_processor, max_length=16, return_indices=True)
            all_tactile_pixel_values.append(pixel_values)
            all_indices.append(torch.tensor(indices, dtype=torch.long))

        if all_tactile_pixel_values:
            stacked_pixel_values = torch.stack(all_tactile_pixel_values, dim=0)
            stacked_indices = torch.stack(all_indices, dim=0)
        else:
            C, H, W = 3, 224, 224
            stacked_pixel_values = torch.empty((0, 16, C, H, W), dtype=torch.float)
            stacked_indices = torch.empty((0, 16), dtype=torch.long)

        return question, answer_tokens, stacked_pixel_values, tactile_paths, question_type, question_step, stacked_indices
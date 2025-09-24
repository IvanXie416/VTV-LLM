import os
import random
import json
import argparse
from constants_video import TRAIN_OBJECTS, VAL_OBJECTS, TEST_OBJECTS


def get_samples(dataset_path, train_json_path, val_json_path, test_json_path):

    random.shuffle(TRAIN_OBJECTS)
    

    dataset_files = os.listdir(dataset_path)
    if '.DS_Store' in dataset_files:
        dataset_files.remove('.DS_Store')
    

    video_files = [f for f in dataset_files if "csv" not in f]
    

    train_sample_paths = {}
    val_sample_paths = {}
    test_sample_paths = {}
    

    for video_file in video_files:

        object_name = "_".join(video_file.split("_")[:-1])
        video_path = os.path.join(dataset_path, video_file)
        

        if len(VAL_OBJECTS) == 0:
            if object_name in TRAIN_OBJECTS:
                rand = random.random()
                if rand < 0.8:  
                    if object_name not in train_sample_paths.keys():
                        train_sample_paths[object_name] = [video_path]
                    else:
                        train_sample_paths[object_name].append(video_path)
                elif rand >= 0.8:  
                    if object_name not in val_sample_paths.keys():
                        val_sample_paths[object_name] = [video_path]
                    else:
                        val_sample_paths[object_name].append(video_path)
        else:
    
            if object_name in TRAIN_OBJECTS:
                if object_name not in train_sample_paths.keys():
                    train_sample_paths[object_name] = [video_path]
                else:
                    train_sample_paths[object_name].append(video_path)
            if object_name in VAL_OBJECTS:
                if object_name not in val_sample_paths.keys():
                    val_sample_paths[object_name] = [video_path]
                else:
                    val_sample_paths[object_name].append(video_path)
                    

        if object_name in TEST_OBJECTS:
            if object_name not in test_sample_paths.keys():
                test_sample_paths[object_name] = [video_path]
            else:
                test_sample_paths[object_name].append(video_path)
    
    with open(train_json_path, 'w') as f:
        json.dump(train_sample_paths, f)
        f.close()
    with open(val_json_path, 'w') as f:
        json.dump(val_sample_paths, f)
        f.close()
    with open(test_json_path, 'w') as f:
        json.dump(test_sample_paths, f)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='vtvllm/vbts_video', help='video directory')
    parser.add_argument('--output_path', default='vtvllm/vbts_video', help='processed samples directory')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    train_json_path = os.path.join(args.output_path, "train_samples.json")
    val_json_path = os.path.join(args.output_path, "val_samples.json")
    test_json_path = os.path.join(args.output_path, "test_samples.json")
    get_samples(args.dataset_path, train_json_path, val_json_path, test_json_path)
    print("Done!")

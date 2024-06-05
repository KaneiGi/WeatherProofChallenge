---

# WeatherProofChallenge

In the [WeatherProof Dataset Challenge: Semantic Segmentation In Adverse Weather (CVPR’24 UG²+)](http://cvpr2023.ug2challenge.org/index.html), we achieved first place. This repository contains training and inference scripts for three models: OneFormer, DepthAnything, and InternImage, all implemented within the MMSegmentation framework.

## Additional Training Dataset

You can download the additional training dataset from [WeatherProofExtra on Hugging Face](https://huggingface.co/datasets/WangFangjun/WeatherProofExtra). This dataset can be used to further train your models for better performance.

## Testing and Inference

To perform inference using the downloaded model weights, you can use the provided `test.sh` script on each model's folder. The six model weights and their results can be found at [WeatherProofChallenge-1st-place on Hugging Face](https://huggingface.co/WangFangjun/WeatherProofChallenge-1st-place).

### test.sh
```bash
#!/usr/bin/env bash

CONFIG=$1 
CHECKPOINT=$2 
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT --show-dir ./result --out output.pkl
```

### Running the test script

1. Download the model weights from the provided link.
2. Save the `test.sh` script in your project directory.
3. Ensure you have a configuration file and the corresponding checkpoint file for the model.

Run the script as follows:
```bash
bash test.sh /path/to/config /path/to/checkpoint
```

This will perform inference using the specified model configuration and checkpoint, saving the results in the `./result` directory and outputting to `output.pkl`.

## Using the Model Merge Script

### Command Line Arguments

- `rgb_path`: Path to the dataset folder containing the images.
- `label_folder`: Path to the folder containing subfolders with the segmentation results from different models.
- `out_label_folder`: Path to the folder where the merged results will be saved.
- `--simplify`: Optional flag to simplify the merging process by using only the first image's result for the remaining images.

### Example

Assume you have the following directory structure:

#### Dataset Folder Structure
```
/path/to/dataset/
    scene1/
        frame_000001.png
        frame_000002.png
        ...
        frame_000299.png
    scene2/
        frame_000001.png
        frame_000002.png
        ...
        frame_000299.png
    ...
```

#### Segmentation Results Folder Structure
```
/path/to/segmentation_results/
    model1/
        scene1/
            frame_000001.png
            frame_000002.png
            ...
            frame_000299.png
        scene2/
            frame_000001.png
            frame_000002.png
            ...
            frame_000299.png
        ...
    model2/
        scene1/
            frame_000001.png
            frame_000002.png
            ...
            frame_000299.png
        scene2/
            frame_000001.png
            frame_000002.png
            ...
            frame_000299.png
        ...
    ...
```

You can run the script as follows:

```bash
python model_merge.py /path/to/dataset /path/to/segmentation_results /path/to/output_results
```

To use the simplified merging process, add the `--simplify` flag:

```bash
python model_merge.py /path/to/dataset /path/to/segmentation_results /path/to/output_results --simplify
```

### Detailed Description

1. **`rgb_path`**: This argument specifies the path to the dataset folder containing the scenes with images that need to be processed. Each scene is a subfolder with images named `frame_000001.png` to `frame_000299.png`.

2. **`label_folder`**: This argument specifies the path to the folder containing subfolders with the segmentation results from different models. Each model's results should be in a separate subfolder, maintaining the same scene and image naming structure as the dataset folder.

3. **`out_label_folder`**: This argument specifies the path to the folder where the merged results will be saved. The output folder structure will mirror the input dataset structure.

4. **`--simplify`**: When this optional flag is included, the script will use only the first image's merged result in each scene to replace the results of the remaining images, significantly speeding up the process.

### Contact

[//]: # (For any questions or issues, please contact [Your Name] at [Your Email].)

---

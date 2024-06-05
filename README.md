

# WeatherProofChallenge

In the [WeatherProof Dataset Challenge: Semantic Segmentation In Adverse Weather (CVPR’24 UG²+)](http://cvpr2023.ug2challenge.org/index.html), we achieved first place. This repository contains training and inference scripts for three models: OneFormer, DepthAnything, and InternImage, all implemented within the MMSegmentation framework.

## Additional Training Dataset

You can download the additional training dataset from [WeatherProofExtra on Hugging Face](https://huggingface.co/datasets/WangFangjun/WeatherProofExtra). This dataset can be used to further train your models for better performance.

## Testing and Inference

To perform inference using the downloaded model weights, you can use the provided script on each model's folder. The six model weights and their results can be found at [WeatherProofChallenge-1st-place on Hugging Face](https://huggingface.co/WangFangjun/WeatherProofChallenge-1st-place).


### Running the InterImage test script

1. Download the model weights from the provided link.
2. Ensure you have a configuration file and the corresponding checkpoint file for the model.
3. run the `test.sh` script in `InterImageClip` directory.

```bash
#!/usr/bin/env bash

CONFIG=$1 
CHECKPOINT=$2 
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT --show-dir ./result --out output.pkl
```

This will perform inference using the specified model configuration and checkpoint, saving the results in the `./result` directory and outputting to `output.pkl`.

### Running the OneFormer test cript

1. Download the model weights from the provided link.
2. Ensure you have the appropriate configuration file for the OneFormer model.
3. Run the `demo.sh` script in `OneFormer` directory:
```bash
CUDA_VISIBLE_DEVICES=0 python demo/demo.py --config-file configs/weatherproof/swin/oneformer_swin_large_bs16_100ep.yaml \
  --input datasets/WeatherProofTest \
  --output datasets/WeatherProofTestOut \
  --task semantic \
  --confidence-threshold 0.5 \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS outputs/weatherproof_swin_large_extra_30/model_0007499.pth
```

This will perform inference using the OneFormer model, saving the results in the specified output directory.
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
/path/to/rgb_path/
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
/path/to/label_folder/
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

### Contact

[//]: # (For any questions or issues, please contact [Your Name] at [Your Email].)

---

# WeatherProofChallenge
Model Merge Script
This script, model_merge.py, merges the segmentation results from multiple models. The script allows you to specify the dataset folder, the folder containing the prediction results from each model, and the folder to save the merged results. Additionally, it provides an option to simplify the merging process by using only the first image's merged result to replace the remaining images, speeding up the computation.

Requirements
Python 3.x
OpenCV
NumPy
Installation
First, ensure you have the necessary libraries installed. You can do this using pip:

pip install opencv-python numpy
Usage
Command Line Arguments
rgb_path: Path to the dataset folder containing the images.
label_folder: Path to the folder containing subfolders with the segmentation results from different models.
out_label_folder: Path to the folder where the merged results will be saved.
--simplify: Optional flag to simplify the merging process by using only the first image's result for the remaining images.
Example
Assume you have the following directory structure:

Dataset Folder Structure
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
Segmentation Results Folder Structure
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
You can run the script as follows:

python model_merge.py /path/to/dataset /path/to/segmentation_results /path/to/output_results
To use the simplified merging process, add the --simplify flag:

python model_merge.py /path/to/dataset /path/to/segmentation_results /path/to/output_results --simplify
Detailed Description
rgb_path: This argument specifies the path to the dataset folder containing the scenes with images that need to be processed. Each scene is a subfolder with images named frame_000001.png to frame_000299.png.

label_folder: This argument specifies the path to the folder containing subfolders with the segmentation results from different models. Each model's results should be in a separate subfolder, maintaining the same scene and image naming structure as the dataset folder.

out_label_folder: This argument specifies the path to the folder where the merged results will be saved. The output folder structure will mirror the input dataset structure.

--simplify: When this optional flag is included, the script will use only the first image's merged result in each scene to replace the results of the remaining images, significantly speeding up the process.

Script Workflow
The script reads the scenes and images from the specified dataset folder (rgb_path).
It then reads the segmentation results from each model's subfolder within the specified label folder (label_folder).
The script merges these segmentation results using predefined weights.
If the --simplify flag is used, the merged result of the first image in each scene is used for all subsequent images within that scene.
The final merged results are saved to the specified output folder (out_label_folder), maintaining the same scene and image naming structure.
Contact

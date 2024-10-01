# Anomaly_Detection

## Project Overview

This project implements an **Anomaly Detection** system for video frames using a **Convolutional Autoencoder** model. The model aims to detect anomalies in pedestrian footage from the **UCSD Anomaly Detection Dataset**. It reconstructs the input frames and identifies anomalies based on reconstruction errors. The project uses a pre-trained **VGG16** model as a feature extractor to enhance performance.

## Dataset

*The folders Peds1 and Peds2 contain the individual frames of each clip in TIFF format.

*Peds1 contains 34 training video samples and 36 testing video samples. 

*Peds2 contains 16 training video samples and 12 testing video samples. All testing samples are associated with a manually-collected frame-level abnormal events annotation ground truth list (in the .m file).
   
*The training clips in both sets contain ONLY NORMAL FRAMES. 

*Each of the testing clips contain AT LEAST SOME ANOMALOUS FRAMES. 
A frame-level annotation of abnormal events is provided in the ground truth list under the test folder (in the form of a MATLAB .m file). The field 'gt_frame' indicates frames that contain abnormal events.

*10 test clips from the Peds1 set and 12 from Ped2 are also provided with PIXEL LEVEL GROUNDTRUTH MASKS.
These masks are labeled "Test004_gt", "Test014_gt", "Test018_gt" etc. in the Peds1 folder. 
(There is also full pixel level annotation on Ped1 for all 36 testing videos available at http://hci.iwr.uni-heidelberg.de/COMPVIS/research/abnormality)
# Anomaly Detection in Video Frames using Autoencoders

### Directory Structure
- `Train`: Contains training clips of pedestrian activity.
- `Test`: Contains testing clips with both normal and anomalous events.

## Key Features
- **Autoencoder Architecture**: Utilizes a Convolutional Autoencoder with VGG16 as a feature extractor for improved representation learning.
- **Anomaly Mask Generation**: Generates binary masks indicating areas of anomalies based on reconstruction errors.
- **Dynamic Thresholding**: Anomalies are detected using a dynamic threshold based on reconstruction error percentiles.
- **Evaluation Metrics**: The model's performance is evaluated using **F1 Score**.

## Model Architecture

The model consists of the following components:
1. **Encoder**: Uses the VGG16 model for feature extraction from input images.
2. **Decoder**: Reconstructs images from the encoded features using upsampling and convolutional layers.
3. **Loss Function**: Mean Squared Error (MSE) is used to measure the reconstruction quality.

### Training
- The model is trained using the training set for **30 epochs** with a batch size of **32**.
- The input images are normalized to the range [0, 1].

## Code Overview

### Functions
- `load_images_from_folder(folder)`: Loads and preprocesses images from the specified folder.
- `load_ground_truth_masks(folder)`: Loads and preprocesses ground truth masks.
- `generate_anomaly_masks(original, reconstructed)`: Generates anomaly masks based on reconstruction errors.
- `evaluate_masks(predicted_masks, ground_truth_masks)`: Evaluates predicted masks against ground truth masks using F1 Score.

### Results
The model generates anomaly masks for the test set and evaluates them against ground truth masks. The F1 Score quantifies the model's performance in detecting anomalies.

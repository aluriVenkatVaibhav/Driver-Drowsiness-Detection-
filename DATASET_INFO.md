# Dataset Information

This project is designed for driver drowsiness detection using video data and a trained deep learning model.

## Dataset Purpose

The dataset is used to train or evaluate a model that detects signs of driver fatigue such as:
- closed eyes
- yawning
- head tilting
- prolonged inattentiveness

## Expected Data Format

### Video Data
The application accepts video files for inference, typically in one of the following formats:
- .mp4
- .avi
- .mov

### Model File
A trained Keras/TensorFlow model is required for prediction:
- .h5

### Optional Audio File
A warning beep can be provided for output videos:
- .mp3

## Recommended Dataset Structure

If you are preparing your own dataset, a simple structure can look like this:

```text
dataset/
├── train/
│   ├── alert/
│   └── drowsy/
├── test/
│   ├── alert/
│   └── drowsy/
```

Each class folder may contain:
- frame images
- short video clips
- extracted facial images

## Suggested Labels

Use labels such as:
- alert
- drowsy
- yawning
- sleepy

## Notes for Training

For better results, ensure your dataset contains:
- varied lighting conditions
- different face angles
- both daytime and nighttime samples
- balanced samples across classes
- clear facial visibility

## Notes for Inference

For this project, the app expects:
- an input video file
- a trained .h5 model file
- optionally, a .mp3 warning sound file

## Example Files

The project already includes:
- drowsiness_new6.h5
- haarcascade_frontalface_default.xml

## Important Recommendation

If you plan to train a new model, use a large and diverse dataset with properly labeled samples to improve detection accuracy.

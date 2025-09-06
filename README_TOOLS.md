# FastVLM Tools and Utilities

This repository includes several utility scripts to make working with FastVLM more efficient:

## 1. Interactive Prediction Script (`predict_interactive.py`)

A faster alternative to the standard [predict.py](file:///D:/Development/GitHub/ml-fastvlm/predict.py) that loads the model once and allows continuous image captioning.

### Features:
- **Interactive Mode**: Enter image paths one by one after the model loads
- **Batch Mode**: Process multiple images from a list file
- **Custom Prompts**: Use different prompts for each image or a single prompt for all images
- **Much faster**: The model is loaded only once, not for each image

### Usage:
```bash
# Interactive mode
python predict_interactive.py --model-path /path/to/checkpoint-dir

# Batch mode
python predict_interactive.py --model-path /path/to/checkpoint-dir --batch-file image_list.txt
```

See [README_INTERACTIVE.md](README_INTERACTIVE.md) for detailed instructions.

## 2. Batch File Creator (`create_batch_file.py`)

Utility script to scan a directory for image files and create a list file for batch processing.

### Usage:
```bash
# Scan current directory
python create_batch_file.py ./images

# Scan recursively and save to custom file
python create_batch_file.py ./images -o my_images.txt -r
```

## 3. Launch Scripts

For easier execution, we provide platform-specific launch scripts:

- `predict_interactive.bat` - Windows batch script
- `predict_interactive.sh` - Bash shell script

These scripts automatically activate the FastVLM conda environment before running the Python script.

## Benefits of Using These Tools

1. **Speed**: Avoid reloading the model for each image (can save 10-30 seconds per image)
2. **Efficiency**: Process multiple images in one session
3. **Flexibility**: Choose between interactive and batch processing modes
4. **Convenience**: Easy-to-use scripts with proper error handling

## Example Workflow

1. Create a batch file of your images:
   ```bash
   python create_batch_file.py ./my_photo_collection -o photos.txt -r
   ```

2. Process all images with a single model load:
   ```bash
   python predict_interactive.py --model-path ./checkpoints/fastvlm-1.5b --batch-file photos.txt
   ```

This approach is significantly faster than running [predict.py](file:///D:/Development/GitHub/ml-fastvlm/predict.py) separately for each image.
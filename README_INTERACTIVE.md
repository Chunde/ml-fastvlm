# FastVLM Interactive Prediction Script

This script allows you to generate captions for multiple images without reloading the model each time, making the process much faster.

## Environment Setup

Make sure you have activated the FastVLM conda environment before running the script:

```bash
conda activate fastvlm
```

If the environment doesn't exist, create it first:

```bash
conda create -n fastvlm python=3.10
conda activate fastvlm
pip install -e .
```

If you encounter any missing dependencies, install them using:

```bash
pip install sympy
# or if you have permission issues
pip install --user sympy
```

## Features

1. **Interactive Mode**: Load the model once and continuously input new images for captioning
2. **Batch Mode**: Process multiple images from a list file
3. **Custom Prompts**: Use different prompts for each image or a single prompt for all images

## Usage

### Interactive Mode

```bash
python predict_interactive.py --model-path /path/to/checkpoint-dir
```

After the model loads, you'll be prompted to enter image paths one by one. The script will generate captions for each image without reloading the model.

### Batch Mode

Create a text file with one image path per line (e.g., `image_list.txt`):

```
/path/to/image1.jpg
/path/to/image2.png
/path/to/image3.jpeg
```

Then run:

```bash
python predict_interactive.py --model-path /path/to/checkpoint-dir --batch-file image_list.txt
```

### Additional Options

- `--prompt`: Specify a custom prompt (default: "Describe the image.")
- `--temperature`: Control randomness in generation (default: 0.2)
- `--num_beams`: Number of beams for beam search (default: 1)

## Benefits

- **Much faster**: The model is loaded only once, not for each image
- **Memory efficient**: Reuses the same model instance
- **Flexible**: Supports both interactive and batch processing
- **Customizable**: Allows different prompts for each image

## Example

```bash
python predict_interactive.py --model-path ./checkpoints/fastvlm-1.5b
```

Enter image paths when prompted, and get captions without waiting for model reloads!
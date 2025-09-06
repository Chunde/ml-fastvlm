# FastVLM GUI Prediction Tool

A graphical user interface for FastVLM image captioning that provides an easy-to-use visual interface.

## Features

1. **Visual Interface**: User-friendly GUI with buttons for file selection
2. **Image Preview**: Displays selected images before processing
3. **Model Management**: Load models through a simple file dialog
4. **Custom Prompts**: Enter custom prompts for each image
5. **Generation Parameters**: Adjustable temperature, top-p, and num beams settings
6. **Real-time Output**: Shows generated captions in a scrollable text area
7. **Threading Support**: Non-blocking operations to keep UI responsive

## Usage

### Running the GUI

```bash
# Direct execution
python predict_gui.py

# Or using the launch scripts
./predict_gui.sh        # Linux/macOS
predict_gui.bat         # Windows
```

### Using the Interface

1. **Load a Model**:
   - Click "Browse..." next to "Model Path" to select your checkpoint directory
   - Click "Load Model" to initialize the model (this may take some time)

2. **Select an Image**:
   - Click "Browse..." next to "Image File" to select an image
   - The image preview will appear in the preview area

3. **Customize Parameters** (optional):
   - Modify the prompt text
   - Adjust generation parameters (temperature, top-p, num beams)

4. **Generate Caption**:
   - Click "Generate Caption" to process the image
   - View the result in the output text area

## Requirements

- Python 3.10+
- FastVLM environment with all dependencies installed
- tkinter (usually included with Python)
- PIL/Pillow for image handling

## Benefits

- **User-Friendly**: No command-line knowledge required
- **Visual Feedback**: Image previews and real-time status updates
- **Efficient**: Model is loaded once and reused for multiple images
- **Customizable**: Full control over generation parameters
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Example Workflow

1. Run the GUI: `python predict_gui.py`
2. Select your model directory (e.g., `./checkpoints/fastvlm-1.5b_stage3`)
3. Click "Load Model" and wait for initialization
4. Select an image file using the "Browse..." button
5. Optionally modify the prompt or generation parameters
6. Click "Generate Caption" to create a description
7. View the result in the output area

The GUI will remain open for processing multiple images with the same model, making it much faster than running separate command-line instances.
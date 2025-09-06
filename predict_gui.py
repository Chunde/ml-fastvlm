#!/usr/bin/env python3
"""
GUI version of FastVLM prediction script
Provides a user-friendly interface for image captioning with FastVLM
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import torch
import argparse

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


class FastVLMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FastVLM Image Captioning")
        self.root.geometry("800x700")
        
        # Model variables
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.model_path = None
        self.conv_mode = "qwen_2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Current image
        self.current_image_path = None
        self.current_image = None
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Model selection
        ttk.Label(main_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_path_var = tk.StringVar()
        model_path_entry = ttk.Entry(main_frame, textvariable=self.model_path_var, width=50)
        model_path_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(0, 5))
        model_browse_btn = ttk.Button(main_frame, text="Browse...", command=self.browse_model)
        model_browse_btn.grid(row=0, column=2, pady=5)
        
        # Load model button
        self.load_model_btn = ttk.Button(main_frame, text="Load Model", command=self.load_model)
        self.load_model_btn.grid(row=1, column=0, columnspan=3, pady=10)
        
        # Image selection
        ttk.Label(main_frame, text="Image File:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.image_path_var = tk.StringVar()
        image_path_entry = ttk.Entry(main_frame, textvariable=self.image_path_var, width=50)
        image_path_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=(0, 5))
        image_browse_btn = ttk.Button(main_frame, text="Browse...", command=self.browse_image)
        image_browse_btn.grid(row=2, column=2, pady=5)
        
        # Prompt input
        ttk.Label(main_frame, text="Prompt:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.prompt_var = tk.StringVar(value="Describe the image.")
        prompt_entry = ttk.Entry(main_frame, textvariable=self.prompt_var, width=50)
        prompt_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5, padx=(0, 5))
        
        # Generation parameters
        params_frame = ttk.LabelFrame(main_frame, text="Generation Parameters", padding="10")
        params_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        params_frame.columnconfigure(1, weight=1)
        
        ttk.Label(params_frame, text="Temperature:").grid(row=0, column=0, sticky=tk.W)
        self.temperature_var = tk.DoubleVar(value=0.2)
        temp_spinbox = ttk.Spinbox(params_frame, from_=0.0, to=1.0, increment=0.1, textvariable=self.temperature_var, width=10)
        temp_spinbox.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(params_frame, text="Top P:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.top_p_var = tk.DoubleVar(value=1.0)
        top_p_spinbox = ttk.Spinbox(params_frame, from_=0.0, to=1.0, increment=0.1, textvariable=self.top_p_var, width=10)
        top_p_spinbox.grid(row=0, column=3, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(params_frame, text="Num Beams:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.num_beams_var = tk.IntVar(value=1)
        beams_spinbox = ttk.Spinbox(params_frame, from_=1, to=10, increment=1, textvariable=self.num_beams_var, width=10)
        beams_spinbox.grid(row=1, column=1, sticky=tk.W, pady=(10, 0), padx=(10, 0))
        
        # Image preview
        self.image_preview_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        self.image_preview_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        self.image_preview_frame.columnconfigure(0, weight=1)
        self.image_preview_frame.rowconfigure(0, weight=1)
        
        self.image_label = ttk.Label(self.image_preview_frame)
        self.image_label.grid(row=0, column=0, padx=10, pady=10)
        
        # Generate button
        self.generate_btn = ttk.Button(main_frame, text="Generate Caption", command=self.generate_caption, state=tk.DISABLED)
        self.generate_btn.grid(row=6, column=0, columnspan=3, pady=10)
        
        # Output text area
        ttk.Label(main_frame, text="Generated Caption:").grid(row=7, column=0, sticky=tk.W, pady=(10, 5))
        self.output_text = scrolledtext.ScrolledText(main_frame, height=10, width=70)
        self.output_text.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Configure weights for resizing
        main_frame.rowconfigure(8, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
    def browse_model(self):
        """Open file dialog to select model directory"""
        model_path = filedialog.askdirectory(title="Select Model Directory")
        if model_path:
            self.model_path_var.set(model_path)
            
    def browse_image(self):
        """Open file dialog to select image file"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
            ("All files", "*.*")
        ]
        image_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=file_types
        )
        if image_path:
            self.image_path_var.set(image_path)
            self.load_image_preview(image_path)
            
    def load_image_preview(self, image_path):
        """Load and display image preview"""
        try:
            # Open and resize image for preview
            image = Image.open(image_path)
            # Resize to fit in preview area (max 300x300)
            image.thumbnail((300, 300))
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep a reference
            self.current_image_path = image_path
            self.current_image = image
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image preview: {str(e)}")
            
    def load_model(self):
        """Load the FastVLM model"""
        model_path = self.model_path_var.get().strip()
        if not model_path:
            messagebox.showerror("Error", "Please select a model path.")
            return
            
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model path does not exist: {model_path}")
            return
            
        # Disable UI during loading
        self.load_model_btn.configure(state=tk.DISABLED, text="Loading...")
        self.root.update()
        
        try:
            # Load model in a separate thread to prevent UI freezing
            def load_model_thread():
                try:
                    # Remove generation config from model folder if it exists
                    generation_config_path = None
                    if os.path.exists(os.path.join(model_path, "generation_config.json")):
                        generation_config_path = os.path.join(model_path, ".generation_config.json")
                        os.rename(os.path.join(model_path, "generation_config.json"), generation_config_path)
                    
                    # Load model
                    disable_torch_init()
                    model_name = get_model_name_from_path(model_path)
                    self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                        model_path, None, model_name, device=self.device
                    )
                    
                    # Set pad token id
                    self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
                    
                    # Store model path
                    self.model_path = model_path
                    self.generation_config_path = generation_config_path
                    
                    # Update UI on main thread
                    self.root.after(0, self.on_model_loaded)
                except Exception as e:
                    self.root.after(0, lambda: self.on_model_load_error(str(e)))
                    
            # Start loading thread
            thread = threading.Thread(target=load_model_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.load_model_btn.configure(state=tk.NORMAL, text="Load Model")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            
    def on_model_loaded(self):
        """Called when model is successfully loaded"""
        self.load_model_btn.configure(state=tk.NORMAL, text="Load Model")
        self.generate_btn.configure(state=tk.NORMAL)
        messagebox.showinfo("Success", "Model loaded successfully!")
        
    def on_model_load_error(self, error_msg):
        """Called when model loading fails"""
        self.load_model_btn.configure(state=tk.NORMAL, text="Load Model")
        messagebox.showerror("Error", f"Failed to load model: {error_msg}")
        
    def generate_caption(self):
        """Generate caption for the selected image"""
        if not self.model:
            messagebox.showerror("Error", "Please load a model first.")
            return
            
        image_path = self.image_path_var.get().strip()
        if not image_path:
            messagebox.showerror("Error", "Please select an image file.")
            return
            
        if not os.path.exists(image_path):
            messagebox.showerror("Error", f"Image file does not exist: {image_path}")
            return
            
        prompt = self.prompt_var.get().strip()
        if not prompt:
            prompt = "Describe the image."
            
        # Disable UI during generation
        self.generate_btn.configure(state=tk.DISABLED, text="Generating...")
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Generating caption...\n")
        self.root.update()
        
        try:
            # Generate caption in a separate thread
            def generate_thread():
                try:
                    result = self.predict(
                        image_path, 
                        prompt,
                        self.temperature_var.get(),
                        self.top_p_var.get(),
                        self.num_beams_var.get()
                    )
                    # Update UI on main thread
                    self.root.after(0, lambda: self.on_caption_generated(result))
                except Exception as e:
                    self.root.after(0, lambda: self.on_generation_error(str(e)))
                    
            # Start generation thread
            thread = threading.Thread(target=generate_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.generate_btn.configure(state=tk.NORMAL, text="Generate Caption")
            messagebox.showerror("Error", f"Failed to generate caption: {str(e)}")
            
    def on_caption_generated(self, result):
        """Called when caption is successfully generated"""
        self.generate_btn.configure(state=tk.NORMAL, text="Generate Caption")
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, result)
        
    def on_generation_error(self, error_msg):
        """Called when caption generation fails"""
        self.generate_btn.configure(state=tk.NORMAL, text="Generate Caption")
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"Error: {error_msg}")
        
    def predict(self, image_file, prompt, temperature=0.2, top_p=None, num_beams=1):
        """Generate caption for an image"""
        # Construct prompt
        qs = prompt
        if self.model.config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        # Tokenize prompt
        input_ids = (
            tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(torch.device(self.device))
        )

        # Load and preprocess image
        image = Image.open(image_file).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]

        # Run inference
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half(),
                image_sizes=[image.size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=512,
                use_cache=True,
            )

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return outputs


def main():
    root = tk.Tk()
    app = FastVLMGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
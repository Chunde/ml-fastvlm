#
# Interactive prediction script for FastVLM
# Loads model once and allows continuous image captioning
#
import os
import argparse
import torch
from PIL import Image

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


class FastVLMInteractive:
    def __init__(self, model_path, model_base=None, conv_mode="qwen_2", device="cuda"):
        self.device = device
        self.conv_mode = conv_mode
        
        # Remove generation config from model folder
        # to read generation parameters from args
        model_path = os.path.expanduser(model_path)
        self.generation_config_path = None
        if os.path.exists(os.path.join(model_path, "generation_config.json")):
            self.generation_config_path = os.path.join(model_path, ".generation_config.json")
            os.rename(os.path.join(model_path, "generation_config.json"), self.generation_config_path)

        # Load model
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name, device=device
        )

        # Set the pad token id for generation
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def predict(self, image_file, prompt="Describe the image."):
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
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=512,
                use_cache=True,
            )

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return outputs

    def cleanup(self):
        # Restore generation config
        if self.generation_config_path is not None:
            original_path = os.path.join(os.path.dirname(self.generation_config_path), "generation_config.json")
            os.rename(self.generation_config_path, original_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch-file", type=str, default=None, help="Path to a text file containing image paths (one per line) for batch processing")
    parser.add_argument("--prompt", type=str, default="Describe the image.", help="Prompt for VLM (used for all images in batch mode)")
    args = parser.parse_args()

    print("Loading model... This may take a while.")
    predictor = FastVLMInteractive(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Model loaded successfully!")

    # Batch mode
    if args.batch_file is not None:
        if not os.path.exists(args.batch_file):
            print(f"Error: Batch file '{args.batch_file}' not found.")
            predictor.cleanup()
            return
            
        with open(args.batch_file, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
            
        print(f"Processing {len(image_paths)} images in batch mode...")
        for i, image_path in enumerate(image_paths):
            if not os.path.exists(image_path):
                print(f"Warning: File '{image_path}' not found. Skipping.")
                continue
                
            try:
                print(f"[{i+1}/{len(image_paths)}] Processing '{image_path}'...")
                result = predictor.predict(image_path, args.prompt)
                print(f"Caption: {result}\n")
            except Exception as e:
                print(f"Error processing '{image_path}': {str(e)}\n")
                
        predictor.cleanup()
        print("Batch processing complete.")
        return

    # Interactive mode
    try:
        while True:
            print("\n" + "="*50)
            image_path = input("Enter the path to an image file (or 'quit' to exit): ").strip()
            
            if image_path.lower() in ['quit', 'exit', 'q']:
                break
                
            if not os.path.exists(image_path):
                print(f"Error: File '{image_path}' not found.")
                continue
                
            try:
                # Try to open the image to verify it's valid
                with Image.open(image_path) as img:
                    img.verify()
            except Exception:
                print("Error: Invalid image file.")
                continue
                
            prompt = input("Enter a prompt (or press Enter for default 'Describe the image.'): ").strip()
            if not prompt:
                prompt = "Describe the image."
                
            try:
                print("Generating caption...")
                result = predictor.predict(image_path, prompt)
                print("\nGenerated Caption:")
                print(result)
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        predictor.cleanup()
        print("Done.")


if __name__ == "__main__":
    main()
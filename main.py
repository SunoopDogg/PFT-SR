import torch
import os
import os.path as osp
import argparse
from PIL import Image

from utils import load_model, process_image, select_patch_settings


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, required=True, help="Input image path.")
    parser.add_argument("-o", "--out_path", type=str, default="results/",
                        help="Output directory path.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument(
        "--task",
        type=str,
        default="classical",
        choices=['classical', 'lightweight'],
        help="Task for the model. classical: for classical SR models. lightweight: for lightweight models."
    )
    args = parser.parse_args()
    return args


def main():
    args = get_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")
    print(f"Input: {args.in_path}")
    print(f"Task: {args.task}, Scale: {args.scale}x")

    # Check input file
    if not os.path.exists(args.in_path):
        print(f"Error: Input file not found: {args.in_path}")
        return

    # Load image
    image = Image.open(args.in_path).convert('RGB')
    width, height = image.size
    print(f"Image size: {width} x {height}")

    # Patch settings GUI
    print("\nOpening patch settings...")
    patch_size = select_patch_settings(image, width, height)

    if patch_size is None:
        print("Patch settings cancelled.")
        return

    print(f"Patch size: {patch_size[0]} x {patch_size[1]}")

    # Load model
    print("\nLoading model...")
    model = load_model(args.task, args.scale, device)
    print("Model loaded.")

    # Create output directory
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # Generate output filename
    file_name = osp.splitext(osp.basename(args.in_path))
    output_filename = f"{file_name[0]}_PFT_{args.task}_SRx{args.scale}{file_name[1]}"
    output_path = os.path.join(args.out_path, output_filename)

    # Process image
    print("\nProcessing...")
    process_image(
        image, output_path,
        model, device, args.scale, patch_size
    )

    print("\nDone!")


if __name__ == "__main__":
    main()

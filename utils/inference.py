import torch
from torchvision import transforms

from .patch_processor import PatchProcessor


def process_image(image, output_path, model, device, scale, patch_size=None):
    """
    Process image to generate SR image

    Args:
        image: PIL Image
        output_path: Output image path
        model: SR model
        device: Device
        scale: Upscale factor
        patch_size: (width, height) or None (process entire image at once)
    """
    with torch.no_grad():
        print(f"Input size: {image.size[0]} x {image.size[1]}")

        # Convert to tensor
        image_input = transforms.ToTensor()(image).unsqueeze(0).to(device)

        # Inference
        if patch_size is None:
            # Process entire image at once
            image_output = model(image_input).clamp(0.0, 1.0)[0].cpu()
        else:
            # Patch-based processing
            print(f"Patch mode: {patch_size[0]} x {patch_size[1]}")
            processor = PatchProcessor(patch_size[0], patch_size[1])
            image_output = processor.process(image_input, model, device, scale)
            image_output = image_output.clamp(0.0, 1.0)[0].cpu()

        # Save result
        image_output = transforms.ToPILImage()(image_output)
        image_output.save(output_path)

        print(f"Output size: {image_output.size[0]} x {image_output.size[1]}")
        print(f"Saved to: {output_path}")

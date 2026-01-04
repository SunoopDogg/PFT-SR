import torch
import torch.nn.functional as F


class PatchProcessor:
    """Splits image into patches for inference and merges results"""

    def __init__(self, patch_width=256, patch_height=256, overlap_ratio=0.1):
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.overlap_ratio = overlap_ratio

    def process(self, image_tensor, model, device, scale):
        """
        Perform patch-based inference

        Args:
            image_tensor: Input image tensor (1, C, H, W)
            model: SR model
            device: Device ('cuda' or 'cpu')
            scale: Upscale factor

        Returns:
            output_tensor: Output image tensor (1, C, H*scale, W*scale)
        """
        _, C, h, w = image_tensor.size()

        # Calculate number of patches
        split_token_h = max(1, h // self.patch_height + (1 if h % self.patch_height else 0))
        split_token_w = max(1, w // self.patch_width + (1 if w % self.patch_width else 0))

        # Calculate padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w

        # Apply padding
        img = F.pad(image_tensor, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()

        # Calculate patch size
        split_h = H // split_token_h
        split_w = W // split_token_w

        # Calculate overlap
        shave_h = int(split_h * self.overlap_ratio)
        shave_w = int(split_w * self.overlap_ratio)

        ral = H // split_h  # number of rows
        row = W // split_w  # number of columns

        # Calculate patch slices
        slices = []
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i * split_h, (i + 1) * split_h + shave_h)
                elif i == ral - 1:
                    top = slice(i * split_h - shave_h, (i + 1) * split_h)
                else:
                    top = slice(i * split_h - shave_h, (i + 1) * split_h + shave_h)

                if j == 0 and j == row - 1:
                    left = slice(j * split_w, (j + 1) * split_w)
                elif j == 0:
                    left = slice(j * split_w, (j + 1) * split_w + shave_w)
                elif j == row - 1:
                    left = slice(j * split_w - shave_w, (j + 1) * split_w)
                else:
                    left = slice(j * split_w - shave_w, (j + 1) * split_w + shave_w)

                slices.append((top, left))

        # Extract patches
        img_chops = []
        for top, left in slices:
            img_chops.append(img[..., top, left])

        total_patches = len(img_chops)
        print(f"Processing {total_patches} patches ({ral}x{row})")

        # Inference on each patch
        outputs = []
        for idx, chop in enumerate(img_chops):
            print(f"  Patch {idx + 1}/{total_patches}", end='\r')
            out = model(chop.to(device))
            outputs.append(out.cpu())

        # Clear cache once after all patches processed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  Patch {total_patches}/{total_patches} - Done")

        # Merge results
        _img = torch.zeros(1, C, H * scale, W * scale)

        for i in range(ral):
            for j in range(row):
                top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                left = slice(j * split_w * scale, (j + 1) * split_w * scale)

                if i == 0:
                    _top = slice(0, split_h * scale)
                else:
                    _top = slice(shave_h * scale, (shave_h + split_h) * scale)

                if j == 0:
                    _left = slice(0, split_w * scale)
                else:
                    _left = slice(shave_w * scale, (shave_w + split_w) * scale)

                _img[..., top, left] = outputs[i * row + j][..., _top, _left]

        # Remove padding
        _, _, h_out, w_out = _img.size()
        output = _img[:, :, 0:h_out - mod_pad_h * scale, 0:w_out - mod_pad_w * scale]

        return output

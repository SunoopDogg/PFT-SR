import torch
from basicsr.archs.pft_arch import PFT


MODEL_PATH = {
    "classical": {
        "2": "experiments/pretrained_models/001_PFT_SRx2_scratch.pth",
        "3": "experiments/pretrained_models/002_PFT_SRx3_finetune.pth",
        "4": "experiments/pretrained_models/003_PFT_SRx4_finetune.pth",
    },
    "lightweight": {
        "2": "experiments/pretrained_models/101_PFT_light_SRx2_scratch.pth",
        "3": "experiments/pretrained_models/102_PFT_light_SRx3_finetune.pth",
        "4": "experiments/pretrained_models/103_PFT_light_SRx4_finetune.pth",
    }
}


def load_model(task, scale, device):
    if task == 'classical':
        model = PFT(
            upscale=scale,
            embed_dim=240,
            depths=[4, 4, 4, 6, 6, 6],
            num_heads=6,
            num_topk=[1024, 1024, 1024, 1024,
                      256, 256, 256, 256,
                      128, 128, 128, 128,
                      64, 64, 64, 64, 64, 64,
                      32, 32, 32, 32, 32, 32,
                      16, 16, 16, 16, 16, 16],
            window_size=32,
            convffn_kernel_size=7,
            mlp_ratio=2,
            upsampler='pixelshuffle',
            use_checkpoint=False,
        )
    elif task == 'lightweight':
        model = PFT(
            upscale=scale,
            embed_dim=52,
            depths=[2, 4, 6, 6, 6],
            num_heads=4,
            num_topk=[1024, 1024,
                      256, 256, 256, 256,
                      128, 128, 128, 128, 128, 128,
                      64, 64, 64, 64, 64, 64,
                      32, 32, 32, 32, 32, 32],
            window_size=32,
            convffn_kernel_size=7,
            mlp_ratio=1,
            upsampler='pixelshuffledirect',
            use_checkpoint=False,
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    state_dict = torch.load(
        MODEL_PATH[task][str(scale)],
        map_location=device,
        weights_only=False
    )['params_ema']
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    return model

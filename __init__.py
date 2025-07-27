from . import WhirlpoolUpscaler
from .WhirlpoolUpscaler import WhirlpoolUpscaler

NODE_CLASS_MAPPINGS = {
    "WhirlpoolUpscaler": WhirlpoolUpscaler
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WhirlpoolUpscaler": "Whirlpool Upscaler"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

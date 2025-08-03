import torch
import comfy.samplers
import comfy.sample
from comfy import model_management
import nodes
import inspect
import math


# Use ComfyUI's native sampling only

# Import ComfyUI interruption handling
try:
    from comfy.model_management import InterruptProcessingException, throw_exception_if_processing_interrupted
    COMFY_INTERRUPT_AVAILABLE = True
except ImportError:
    COMFY_INTERRUPT_AVAILABLE = False

# Constants
LATENT_TO_PIXEL_SCALE = 8
DEFAULT_OVERLAP = 64

# WhirlpoolUpscaler: Progressive Multi-Iteration Sampling with Upscaling
# 
# DETAILED STEP-BY-STEP PROCESS:
#
# 1. INITIALIZATION PHASE:
#    - Parse input parameters (iterations, steps, denoise values, upscale_by, etc.)
#    - Calculate progressive scale factors for each iteration:
#      * Geometric mode: Each iteration multiplies by (upscale_by)^(1/iterations)
#      * Simple mode: Linear increment per iteration: 1.0 + ((iteration + 1) * (upscale_by - 1.0) / iterations)
#    - Calculate progressive denoise values interpolating linearly from denoise_start to denoise_end
#    - Calculate progressive steps values interpolating linearly from steps_start to steps_end
#    - Calculate progressive CFG values interpolating linearly from cfg_start to cfg_end
#    - Ensure image is in proper format [B, H, W, C]
#
# 2. FALLBACK FOR NO VAE:
#    - If VAE is None, performs basic upscaling without sampling
#    - Uses simple image upscaling with the specified method
#    - Returns scaled image
#    - If iterations is 0, returns original image unchanged
#
# 3. MULTI-ITERATION LOOP (for iterations > 0 with VAE):
#    INITIALIZATION: Encode original image to latent space for continuity preservation
#    For each iteration (0 to iterations-1):
#    
#    A. ITERATION SETUP:
#       - Calculate current denoise value, CFG value, and steps for this iteration (all progressive)
#       - Extract current latent dimensions and calculate current pixel dimensions
#       - Calculate target dimensions using pre-computed scale factor for this iteration
#       - Log current iteration info (number, current resolution, target resolution, scale factor, denoise level, steps, CFG)
#    
#    B. UPSCALE AND SAMPLE METHOD:
#       Step 1: Latent upscaling:
#         - VAE decode: latent → pixels
#         - ImageScale: pixels → upscaled_pixels
#         - VAE encode: upscaled_pixels → upscaled_latent
#       Step 2: Sampling:
#         - Use ComfyUI sampling
#         - Apply denoise value for current iteration
#       Step 3: Latent update for next iteration
#    
#    FINALIZATION: Final VAE decode
#
# 4. ERROR HANDLING:
#    - Each VAE operation wrapped in try/catch with fallbacks
#    - On encode failure: attempts smaller tiles, then creates dummy latent
#    - On decode failure: attempts smaller tiles, then creates black image
#    - Final iteration decode failure: attempts fallback decode without scaling
#
# 5. FINAL OUTPUT:
#    - Returns final decoded image from last iteration
#    - If all operations fail, returns scaled version of input image
#
# PROGRESSION CURVE EXAMPLES (4 iterations, applies to resolution, CFG, steps, denoise):
# upscale_curve = 1.0 (linear):     Even progression across iterations
# upscale_curve = 1.27 (geometric): Traditional exponential progression (matches old geometric mode)
# upscale_curve > 1.0: Exponential progression (higher values = more exponential)
# 
# Resolution example (2.0x rescale):
# upscale_curve = 1.0:  1.0x → 1.25x → 1.50x → 1.75x → 2.00x (linear)
# upscale_curve = 1.27: 1.0x → 1.17x → 1.42x → 1.69x → 2.00x (exponential)
#
# MEMORY EFFICIENCY:
# - Progressive upscaling allows working with smaller images initially
# - Tiled VAE operations handle large images without OOM
# - Immediate cleanup of intermediate data
# - Each iteration refines details at progressively higher resolutions

def extract_latent_tensor(latent_data):
    """Helper function to extract tensor from latent data (dict or tensor)."""
    if isinstance(latent_data, dict):
        return latent_data["samples"]
    return latent_data

def ensure_latent_dict(latent_data):
    """Helper function to ensure latent data is in dict format for VAE operations."""
    if isinstance(latent_data, torch.Tensor):
        return {"samples": latent_data}
    return latent_data

def upscale_with_model(upscale_model, image):
    """Upscale image using an AI upscale model - equivalent to ImageUpscaleWithModel"""
    device = model_management.get_torch_device()
    upscale_model.to(device)
    in_img = image.movedim(-1,-3).to(device)
    
    tile = 512
    overlap = 32
    
    oom = True
    while oom:
        try:
            steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
            pbar = comfy.utils.ProgressBar(steps)
            s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
            oom = False
        except model_management.OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                raise e

    upscale_model.cpu()
    s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
    return s

def histogram_match_channel(source, reference, bins=256):
    """Match histogram of source channel to reference channel"""
    # Convert to 0-255 range for histogram calculation
    source_int = (source * 255).clamp(0, 255).long()
    reference_int = (reference * 255).clamp(0, 255).long()
    
    # Calculate histograms
    source_hist = torch.histc(source_int.float(), bins=bins, min=0, max=255)
    reference_hist = torch.histc(reference_int.float(), bins=bins, min=0, max=255)
    
    # Calculate CDFs (cumulative distribution functions)
    source_cdf = torch.cumsum(source_hist, dim=0)
    reference_cdf = torch.cumsum(reference_hist, dim=0)
    
    # Normalize CDFs
    source_cdf = source_cdf / source_cdf[-1]
    reference_cdf = reference_cdf / reference_cdf[-1]
    
    # Create lookup table
    lookup_table = torch.zeros(256, device=source.device)
    
    for i in range(256):
        # Find the closest CDF value in reference
        diff = torch.abs(reference_cdf - source_cdf[i])
        closest_idx = torch.argmin(diff)
        lookup_table[i] = closest_idx.float()
    
    # Apply lookup table
    matched = lookup_table[source_int] / 255.0
    
    return matched

def apply_fix_vae_color(current_image, reference_image, num_samples=1000):
    """
    Apply histogram matching color correction to current_image based on reference_image.
    Matches the color distribution of each RGB channel independently.
    """
    if reference_image is None:
        return current_image
    
    batch_size, height, width, channels = current_image.shape
    ref_height, ref_width = reference_image.shape[1], reference_image.shape[2]
    
    # Resize reference image to match current image size for proper comparison
    if ref_height != height or ref_width != width:
        reference_resized = nodes.ImageScale().upscale(reference_image, "lanczos", width, height, crop="disabled")[0]
    else:
        reference_resized = reference_image
    
    # Apply histogram matching to each channel independently
    corrected_image = current_image.clone()
    
    for channel in range(3):
        source_channel = current_image[0, :, :, channel].flatten()
        reference_channel = reference_resized[0, :, :, channel].flatten()
        
        matched_channel = histogram_match_channel(source_channel, reference_channel)
        corrected_image[0, :, :, channel] = matched_channel.view(height, width)
    
    # Clamp to valid range [0, 1]
    corrected_image = torch.clamp(corrected_image, 0.0, 1.0)
    
    return corrected_image

def vae_decode_tiled(vae, samples, use_tile=True, tile_size=512, overlap=DEFAULT_OVERLAP):
    """VAE decode with tiling support"""
    if use_tile:
        decoder = nodes.VAEDecodeTiled()
        if 'overlap' in inspect.signature(decoder.decode).parameters:
            pixels = decoder.decode(vae, samples, tile_size, overlap=overlap)[0]
        else:
            pixels = decoder.decode(vae, samples, tile_size)[0]
    else:
        pixels = nodes.VAEDecode().decode(vae, samples)[0]
    return pixels

def vae_encode_tiled(vae, pixels, use_tile=True, tile_size=512, overlap=DEFAULT_OVERLAP):
    """VAE encode with tiling support"""
    if use_tile:
        encoder = nodes.VAEEncodeTiled()
        if 'overlap' in inspect.signature(encoder.encode).parameters:
            samples = encoder.encode(vae, pixels, tile_size, overlap=overlap)[0]
        else:
            samples = encoder.encode(vae, pixels, tile_size)[0]
    else:
        samples = nodes.VAEEncode().encode(vae, pixels)[0]
    return samples

def latent_upscale_on_pixel_space(samples, resize_filter, w, h, vae, use_tile=True, tile_size=512, overlap=DEFAULT_OVERLAP, reference_image=None, fix_vae_color_enabled=False, upscale_model=None):
    """Latent upscaling via pixel space conversion with optional AI upscale model"""
    # Ensure samples is in the proper format for VAE decode
    samples_dict = ensure_latent_dict(samples)
    
    # Step 1: VAE decode
    pixels = vae_decode_tiled(vae, samples_dict, use_tile, tile_size, overlap)
    
    # Step 2: Image upscale
    if upscale_model is not None:
        # Use AI upscaling model first, then resize to exact target dimensions
        target_w = int(w)
        target_h = int(h)
        current_w = pixels.shape[2]
        current_h = pixels.shape[1]
        
        # Apply model upscaling iteratively until we reach or exceed target size
        while current_w < target_w or current_h < target_h:
            pixels = upscale_with_model(upscale_model, pixels)
            new_w = pixels.shape[2]
            new_h = pixels.shape[1]
            
            # Check if model didn't actually upscale (1x model)
            if new_w == current_w and new_h == current_h:
                print(f"[WhirlpoolUpscaler] 1x upscale model detected, breaking upscale loop")
                break
            
            current_w = new_w
            current_h = new_h
        
        # Resize to exact target dimensions if needed
        if current_w != target_w or current_h != target_h:
            pixels = nodes.ImageScale().upscale(pixels, resize_filter, target_w, target_h, False)[0]
    else:
        # Use standard image scaling
        pixels = nodes.ImageScale().upscale(pixels, resize_filter, int(w), int(h), False)[0]
    
    # Step 3: Apply color correction if enabled (after upscaling, before VAE encode)
    if fix_vae_color_enabled and reference_image is not None:
        pixels = apply_fix_vae_color(pixels, reference_image)
    
    # Step 4: VAE encode  
    upscaled_latent = vae_encode_tiled(vae, pixels, use_tile, tile_size, overlap)
    
    return upscaled_latent, pixels


@torch.no_grad()
def lanczos_upscale(image, resize_filter="lanczos", target_height=None, target_width=None, scale_factor=1.0):
    """Upscale image using ComfyUI's ImageScale node with proper Lanczos support."""
    if scale_factor == 1.0 and target_height is None and target_width is None:
        return image
    
    # Ensure image is in [B, H, W, C] format for ImageScale
    if len(image.shape) == 4 and image.shape[1] in [1, 3]:
        image = image.permute(0, 2, 3, 1)
    
    batch_size, height, width, channels = image.shape
    
    if target_height is not None and target_width is not None:
        new_height = target_height
        new_width = target_width
    else:
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
    
    # Use ComfyUI's native ImageScale node for proper Lanczos upscaling
    upscaled = nodes.ImageScale().upscale(image, resize_filter, new_width, new_height, crop="disabled")[0]
    
    return upscaled


def common_upscaler(model, seed, steps_start, steps_end, cfg_start, cfg_end, sampler_name, scheduler, positive, negative, image, denoise_start=1.0, vae=None, iterations=4, denoise_end=0.05, upscale_by=1.0, resize_filter="lanczos", upscale_curve=1.0, tile_size=512, add_noise=0.0, fix_vae_color=False, upscale_model_opt=None):
    # Ensure image is in [B, H, W, C] format
    if len(image.shape) == 4 and image.shape[1] in [1, 3]:
        image = image.permute(0, 2, 3, 1)
    
    # If iterations is 0, return original image unchanged
    if iterations == 0:
        return image
    
    # If VAE is not provided, use simple upscaling
    if vae is None:
        # For no VAE, just do simple upscaling
        target_width = int(image.shape[2] * upscale_by)
        target_height = int(image.shape[1] * upscale_by)
        return lanczos_upscale(image, resize_filter, target_height=target_height, target_width=target_width)
    
    # Calculate scaling parameters based on upscale_curve
    original_width = image.shape[2]
    original_height = image.shape[1]
    
    # PROGRESSION PATTERNS:
    # - Resolution scaling: Each iteration makes actual upscaling progress (1.129x → 1.311x → 1.520x → 1.750x)
    # - Parameters: Start with the start values and progress to end values (steps: 20 → 16 → 10 → 4)
    # This requires different t calculations for resolution vs parameters
    
    # Pre-calculate all scale factors for all iterations
    scale_factors = []
    if iterations == 1:
        # For single iteration, use the final upscale factor
        scale_factors = [upscale_by]
    elif iterations > 1:
        for i in range(iterations):
            # Calculate progress through iterations (1/iterations to 1.0)
            # This ensures each iteration does actual upscaling
            t = (i + 1) / iterations
            
            # Apply upscale_curve to the progression
            # upscale_curve = 1.0: linear, >1.0: exponential, <1.0: inverse exponential
            curved_t = t ** upscale_curve
            current_scale = 1.0 + curved_t * (upscale_by - 1.0)
            
            scale_factors.append(current_scale)
    
    # Pre-calculate all denoise values
    denoise_values = []
    if iterations == 1:
        # For single iteration, prioritize denoise_end if different from denoise_start
        denoise_values = [denoise_end if denoise_end != denoise_start else denoise_start]
    else:
        for i in range(iterations):
            t = i / (iterations - 1)
            # Apply upscale_curve to the progression
            # upscale_curve = 1.0: linear, >1.0: exponential, <1.0: inverse exponential
            curved_t = t ** upscale_curve
            current_denoise = denoise_start + curved_t * (denoise_end - denoise_start)
            denoise_values.append(current_denoise)
    
    # Pre-calculate all steps values
    steps_values = []
    if iterations == 1:
        # For single iteration, prioritize steps_end if different from steps_start
        steps_values = [steps_end if steps_end != steps_start else steps_start]
    else:
        for i in range(iterations):
            t = i / (iterations - 1)
            # Apply upscale_curve to the progression
            # upscale_curve = 1.0: linear, >1.0: exponential, <1.0: inverse exponential
            curved_t = t ** upscale_curve
            current_steps = int(steps_start + curved_t * (steps_end - steps_start))
            steps_values.append(current_steps)
    
    # Pre-calculate all CFG values
    cfg_values = []
    if iterations == 1:
        # For single iteration, prioritize cfg_end if different from cfg_start
        cfg_values = [cfg_end if cfg_end != cfg_start else cfg_start]
    else:
        for i in range(iterations):
            t = i / (iterations - 1)
            # Apply upscale_curve to the progression
            # upscale_curve = 1.0: linear, >1.0: exponential, <1.0: inverse exponential
            curved_t = t ** upscale_curve
            current_cfg = cfg_start + curved_t * (cfg_end - cfg_start)
            cfg_values.append(current_cfg)
    
    
    current_seed = seed
    
    # Encode original image to latent space once
    # This preserves the original image characteristics throughout iterations
    original_latent = vae_encode_tiled(vae, image, use_tile=True, tile_size=tile_size, overlap=DEFAULT_OVERLAP)
    current_latent = original_latent
    
    # For color correction tracking - always use original image as reference
    reference_image = image if fix_vae_color else None
    
    for iteration in range(iterations):
        current_denoise = denoise_values[iteration]
        current_cfg = cfg_values[iteration]
        current_steps = steps_values[iteration]
        # Calculate noise as multiplier of current denoise value
        current_noise = current_denoise * add_noise
        
        # Get current and target resolutions for logging (from latent dimensions)
        # Extract tensor from latent dict for shape calculation
        current_latent_tensor = extract_latent_tensor(current_latent)
        current_latent_height = current_latent_tensor.shape[2]
        current_latent_width = current_latent_tensor.shape[3]
        current_height = current_latent_height * LATENT_TO_PIXEL_SCALE
        current_width = current_latent_width * LATENT_TO_PIXEL_SCALE
        current_scale = scale_factors[iteration]
        target_width = int(original_width * current_scale)
        target_height = int(original_height * current_scale)
        
        # Log iteration information
        if add_noise > 0.0:
            print(f"[WhirlpoolUpscaler] Iteration {iteration + 1}/{iterations} | Current: {current_width}x{current_height} | Target: {target_width}x{target_height} | Scale: {current_scale:.3f}x | Denoise: {current_denoise:.3f} | Steps: {current_steps} | CFG: {current_cfg:.1f} | Add Noise: {current_noise:.3f} (×{add_noise})")
        else:
            print(f"[WhirlpoolUpscaler] Iteration {iteration + 1}/{iterations} | Current: {current_width}x{current_height} | Target: {target_width}x{target_height} | Scale: {current_scale:.3f}x | Denoise: {current_denoise:.3f} | Steps: {current_steps} | CFG: {current_cfg:.1f}")
        
        try:
            # Upscale and sample method
            
            # Memory management
            model_management.soft_empty_cache()
            
            # STEP 1: Latent upscaling via pixel space
            upscaled_latent, upscaled_pixels = latent_upscale_on_pixel_space(
                current_latent, resize_filter, target_width, target_height, vae, 
                use_tile=True, tile_size=tile_size, overlap=DEFAULT_OVERLAP,
                reference_image=reference_image, fix_vae_color_enabled=fix_vae_color,
                upscale_model=upscale_model_opt
            )
            
            # STEP 2: Sampling
            
            latent_tensor = extract_latent_tensor(upscaled_latent)
            
            # Add noise before denoising if noise level > 0
            if current_noise > 0.0:
                # Generate noise tensor with same shape as latent
                additional_noise = torch.randn_like(latent_tensor) * current_noise
                latent_tensor = latent_tensor + additional_noise
            
            noise = comfy.sample.prepare_noise(latent_tensor, current_seed, None)
            
            refined_samples = comfy.sample.sample(
                model, noise, current_steps, current_cfg, sampler_name, scheduler, positive, negative, latent_tensor,
                denoise=current_denoise, disable_noise=False, start_step=None, last_step=None,
                force_full_denoise=True, noise_mask=None, seed=current_seed
            )
            refined_latent = ensure_latent_dict(refined_samples)
            
            # STEP 3: Update current latent for next iteration
            current_latent = refined_latent
            
            # Clean up intermediate data
            del upscaled_latent, upscaled_pixels
            model_management.soft_empty_cache()
            
        except KeyboardInterrupt:
            # Re-raise cancellation/interrupt signals to properly cancel the entire operation
            raise
        except Exception as e:
            # Handle ComfyUI's specific interruption exception
            if COMFY_INTERRUPT_AVAILABLE and isinstance(e, InterruptProcessingException):
                raise
            
            # Check if this is a ComfyUI cancellation exception
            exception_str = str(e).lower()
            exception_type = type(e).__name__.lower()
            if ("interrupted" in exception_str or "cancelled" in exception_str or "cancel" in exception_str or
                "interrupted" in exception_type or "cancelled" in exception_type or "cancel" in exception_type or
                "keyboardinterrupt" in exception_type):
                raise  # Re-raise cancellation exceptions
            
            print(f"Warning: Iteration {iteration + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            # If iteration fails, continue with current latent (fallback)
            print(f"[WhirlpoolUpscaler] Iteration {iteration + 1} failed, using fallback")
            # For final iteration, try simple upscaling as fallback
            if iteration == iterations - 1:
                try:
                    # Try to decode and upscale as last resort
                    fallback_image = vae_decode_tiled(vae, current_latent, use_tile=True, tile_size=tile_size)
                    fallback_upscaled = lanczos_upscale(fallback_image, resize_filter, target_height=target_height, target_width=target_width)
                    # Re-encode for consistency
                    current_latent = vae_encode_tiled(vae, fallback_upscaled, use_tile=True, tile_size=tile_size)
                except KeyboardInterrupt:
                    raise
                except Exception as final_e:
                    # Handle ComfyUI interruption in fallback
                    if COMFY_INTERRUPT_AVAILABLE and isinstance(final_e, InterruptProcessingException):
                        raise
                    
                    # Check for cancellation in fallback too
                    if ("interrupted" in str(final_e).lower() or "cancelled" in str(final_e).lower() or
                        "cancel" in str(final_e).lower()):
                        raise
                    
                    print(f"Final fallback failed: {final_e}")
                    # Keep current latent as-is
        

    # Final decode: Convert final latent back to image space
    try:
        final_image = vae_decode_tiled(vae, current_latent, use_tile=True, tile_size=tile_size, overlap=DEFAULT_OVERLAP)
        
        # Apply final color correction to the output image
        if fix_vae_color and reference_image is not None:
            final_image = apply_fix_vae_color(final_image, reference_image)
        
        return final_image
    except Exception as e:
        print(f"Final decode failed: {e}, using fallback decode")
        try:
            # Fallback to simpler decode
            final_image = vae_decode_tiled(vae, current_latent, use_tile=False)
            
            # Apply final color correction to fallback image too
            if fix_vae_color and reference_image is not None:
                final_image = apply_fix_vae_color(final_image, reference_image)
            
            return final_image
        except Exception as e2:
            print(f"All decode methods failed: {e2}, returning black image")
            # Ultimate fallback - return black image at target size
            target_scale = scale_factors[-1] if scale_factors else 1.0
            final_height = int(original_height * target_scale)
            final_width = int(original_width * target_scale)
            return torch.zeros((1, final_height, final_width, 3), dtype=torch.float32, device="cpu")

class WhirlpoolUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "upscale_by": ("FLOAT", {"default": 2.00, "min": 0.1, "max": 10000, "step": 0.01, "tooltip": "Final resolution multiplier (e.g., 2.0 = double width and height)."}),
                "upscale_curve": ("FLOAT", {"default": 1.75, "min": 1.0, "max": 10.0, "step": 0.01, "tooltip": "Progression curve for all parameters (resolution, CFG, steps, denoise). 1.0 = linear progression, >1.0 = exponential progression (higher values = more exponential)."}),
                "iterations": ("INT", {"default": 4, "min": 0, "max": 20, "tooltip": "Number of complete sampling cycles to perform."}),
                "steps_start": ("INT", {"default": 24, "min": 1, "max": 10000, "tooltip": "Number of sampling steps for the first iteration."}),
                "steps_end": ("INT", {"default": 2, "min": 1, "max": 10000, "tooltip": "Number of sampling steps for the last iteration."}),
                "cfg_start": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01, "tooltip": "CFG scale for the first iteration."}),
                "cfg_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01, "tooltip": "CFG scale for the last iteration."}),
                "denoise_start": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise strength for the first iteration."}),
                "denoise_end": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise strength for the last iteration."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler_ancestral"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "sgm_uniform"}),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "resize_filter": (["lanczos", "nearest-exact", "bilinear", "area", "bicubic"], {"default": "lanczos", "tooltip": "Image resizing filter algorithm."}),
                "tile_size": ("INT", {"default": 1024, "min": 320, "max": 2048, "step": 64, "tooltip": "Tile size for VAE operations."}),
                "vae": ("VAE", ),
                "add_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Adds noise before denoising each iteration. The amount added is relative to the current denoise."}),
                "fix_vae_color": ("BOOLEAN", {"default": True, "tooltip": "Apply color correction after each iteration to maintain color consistency with the original image."}),
            },
            "optional": {
                "upscale_model_opt": ("UPSCALE_MODEL", {"tooltip": "Optional AI upscaling model (e.g., ESRGAN, Real-ESRGAN) for enhanced image quality during upscaling."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"

    CATEGORY = "Whirlpool Upscaler"

    def sample(self, model, seed, iterations, steps_start, steps_end, cfg_start, cfg_end, sampler_name, scheduler, positive, negative, image, denoise_start=1.0, denoise_end=1.0, add_noise=0.0, fix_vae_color=False, upscale_by=1.0, resize_filter="lanczos", upscale_curve=1.0, tile_size=512, vae=None, upscale_model_opt=None):
        image_out = common_upscaler(model, seed, steps_start, steps_end, cfg_start, cfg_end, sampler_name, scheduler, positive, negative, image, denoise_start=denoise_start, vae=vae, iterations=iterations, denoise_end=denoise_end, upscale_by=upscale_by, resize_filter=resize_filter, upscale_curve=upscale_curve, tile_size=tile_size, add_noise=add_noise, fix_vae_color=fix_vae_color, upscale_model_opt=upscale_model_opt)
        
        if image_out is None:
            # Create a black image if everything fails
            # Get the shape from the input image and rescale factor
            height = int(image.shape[1] * upscale_by)
            width = int(image.shape[2] * upscale_by)
            image_out = torch.zeros((image.shape[0], height, width, 3), dtype=torch.float32, device="cpu")

        return (image_out,) 
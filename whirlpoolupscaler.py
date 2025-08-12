import torch
import comfy.samplers
import comfy.sample
from comfy import model_management
import nodes
import inspect
import math
import time
import comfy_extras.nodes_upscale_model as model_upscale
import cv2
import numpy as np
from scipy import ndimage


# Use ComfyUI's native sampling only

# Import ComfyUI interruption handling
try:
    from comfy.model_management import InterruptProcessingException, throw_exception_if_processing_interrupted
    COMFY_INTERRUPT_AVAILABLE = True
except ImportError:
    COMFY_INTERRUPT_AVAILABLE = False

# Constants
PRINT_DEBUG_MESSAGES = False
LATENT_TO_PIXEL_SCALE = 8
DEFAULT_OVERLAP = 64

# Global timer for debug messages
_debug_start_time = None

def get_debug_time():
    """Get elapsed time in milliseconds since debug timer started"""
    global _debug_start_time
    if _debug_start_time is None:
        _debug_start_time = time.time()
        return 0
    return int((time.time() - _debug_start_time) * 1000)

def reset_debug_timer():
    """Reset the debug timer"""
    global _debug_start_time
    _debug_start_time = time.time()

# WhirlpoolUpscaler: Progressive Multi-Iteration Sampling with Upscaling
# 
# DETAILED STEP-BY-STEP PROCESS:
#
# 1. INITIALIZATION PHASE:
#    - Parse input parameters (iterations, steps, denoise values, upscale_by, etc.)
#    - Calculate progressive scale factors for each iteration using upscale_curve:
#      * Each iteration scale: 1.0 + curved_t * (upscale_by - 1.0)
#      * curved_t = t^upscale_curve where t = (iteration + 1) / iterations
#      * upscale_curve = 1.0: linear progression, >1.0: exponential progression
#    - Calculate progressive denoise values using upscale_curve from denoise_start to denoise_end
#    - Calculate progressive steps values using upscale_curve from steps_start to steps_end
#    - Calculate progressive CFG values using upscale_curve from cfg_start to cfg_end
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
#       Step 1: Latent upscaling via pixel space:
#         - VAE decode: latent → pixels
#         - Image upscaling: pixels → upscaled_pixels (using ImageScale or AI upscale model)
#         - VAE encode: upscaled_pixels → upscaled_latent
#       Step 2: Color correction (if enabled):
#         - Apply LAB histogram color matching to upscaled pixels using original image as reference
#         - Re-encode color-corrected pixels back to latent space
#       Step 3: Sampling:
#         - Add noise if add_noise > 0 (amount = current_denoise * add_noise)
#         - Use ComfyUI sampling with current iteration's denoise, CFG, and steps
#       Step 4: Latent update for next iteration
#    
#    FINALIZATION: Final VAE decode with optional color correction
#
# 4. ERROR HANDLING:
#    - Each VAE operation wrapped in try/catch with fallbacks
#    - On encode failure: attempts smaller tiles, then creates dummy latent
#    - On decode failure: attempts smaller tiles, then creates black image
#    - Final iteration decode failure: attempts fallback decode without scaling
#
# 5. FINAL OUTPUT:
#    - Final VAE decode: Convert final latent back to image space
#    - Apply LAB histogram color correction if fix_vae_color is enabled
#    - Returns final decoded image with optional color correction
#    - If decode fails: attempts fallback decode without tiling, still applies color correction
#    - If all operations fail, returns black image at target size
#
# PROGRESSION CURVE EXAMPLES (4 iterations, applies to resolution, CFG, steps, denoise):
# upscale_curve = 1.0 (linear):     Even progression across iterations
# upscale_curve = 3.25 (default):   Exponential progression with more aggressive early steps
# upscale_curve > 1.0: Exponential progression (higher values = more exponential)
# 
# Resolution example (2.0x rescale, 4 iterations):
# upscale_curve = 1.0:  1.0x → 1.25x → 1.50x → 1.75x → 2.00x (linear)
# upscale_curve = 3.25: 1.0x → 1.13x → 1.45x → 1.77x → 2.00x (exponential)
# 
# COLOR CORRECTION:
# - Uses LAB histogram matching for perceptually accurate color transfer
# - Applied after each iteration's upscaling and after final decode
# - Always uses original image as reference for color consistency
# - Transfer strength hardcoded to 1.0 for maximum accuracy
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

def upscale_with_model(upscale_model, image, upscale_decode_size=512):
    """Upscale image using ComfyUI's ImageUpscaleWithModel - with built-in OOM handling"""
    if PRINT_DEBUG_MESSAGES:
        print(f"[WhirlpoolUpscaler DEBUG] Model Upscale: input shape {image.shape}, model_scale={upscale_model.scale} [{get_debug_time()}ms]")
    
    # Check for interruption before attempting upscaling
    if COMFY_INTERRUPT_AVAILABLE:
        throw_exception_if_processing_interrupted()
    
    # Use ComfyUI's built-in ImageUpscaleWithModel with automatic tiling and OOM handling
    upscaled = model_upscale.ImageUpscaleWithModel().upscale(upscale_model, image)[0]
    
    if PRINT_DEBUG_MESSAGES:
        print(f"[WhirlpoolUpscaler DEBUG] Model Upscale complete: output shape {upscaled.shape} [{get_debug_time()}ms]")
    
    return upscaled



def rgb_to_lab(rgb):
    """Convert RGB to LAB color space for better color matching"""
    # Convert RGB [0,255] to [0,1]
    rgb_norm = rgb / 255.0
    
    # Apply gamma correction (sRGB to linear RGB)
    mask = rgb_norm > 0.04045
    rgb_linear = np.where(mask, ((rgb_norm + 0.055) / 1.055) ** 2.4, rgb_norm / 12.92)
    
    # RGB to XYZ transformation matrix (sRGB D65)
    xyz_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    # Convert to XYZ
    xyz = np.dot(rgb_linear, xyz_matrix.T)
    
    # XYZ to LAB conversion
    # D65 illuminant
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    
    fx = xyz[:, :, 0] / xn
    fy = xyz[:, :, 1] / yn
    fz = xyz[:, :, 2] / zn
    
    # Apply cube root transformation
    delta = 6.0 / 29.0
    mask = np.array([fx, fy, fz]) > delta ** 3
    fx = np.where(mask[0], np.cbrt(fx), (fx / (3 * delta ** 2)) + (4.0 / 29.0))
    fy = np.where(mask[1], np.cbrt(fy), (fy / (3 * delta ** 2)) + (4.0 / 29.0))
    fz = np.where(mask[2], np.cbrt(fz), (fz / (3 * delta ** 2)) + (4.0 / 29.0))
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return np.stack([L, a, b], axis=-1)

def lab_to_rgb(lab):
    """Convert LAB to RGB color space"""
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    
    # LAB to XYZ
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    delta = 6.0 / 29.0
    
    # Apply inverse cube root transformation
    mask_x = fx > delta
    mask_y = fy > delta
    mask_z = fz > delta
    
    x = np.where(mask_x, fx ** 3, 3 * delta ** 2 * (fx - 4.0 / 29.0))
    y = np.where(mask_y, fy ** 3, 3 * delta ** 2 * (fy - 4.0 / 29.0))
    z = np.where(mask_z, fz ** 3, 3 * delta ** 2 * (fz - 4.0 / 29.0))
    
    # D65 illuminant
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    xyz = np.stack([x * xn, y * yn, z * zn], axis=-1)
    
    # XYZ to RGB transformation matrix (sRGB D65)
    rgb_matrix = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    
    # Convert to linear RGB
    rgb_linear = np.dot(xyz, rgb_matrix.T)
    
    # Apply gamma correction (linear RGB to sRGB)
    mask = rgb_linear > 0.0031308
    # Clamp to avoid invalid values in power operation
    rgb_linear_safe = np.clip(rgb_linear, 0.0, None)
    rgb_norm = np.where(mask, 1.055 * (rgb_linear_safe ** (1.0 / 2.4)) - 0.055, 12.92 * rgb_linear_safe)
    
    # Convert to [0,255] and clip
    rgb = np.clip(rgb_norm * 255.0, 0, 255)
    return rgb.astype(np.uint8)

def lab_histogram_color_transfer(source, reference, sigma=1.2, transfer_strength=1.0):
    """
    LAB histogram matching color transfer for accurate color correction.
    """
    source_float = source.astype(np.float32)
    reference_float = reference.astype(np.float32)
    
    # Convert to LAB color space
    source_lab = rgb_to_lab(source_float)
    reference_lab = rgb_to_lab(reference_float)
    
    result_lab = source_lab.copy()
    
    # Apply histogram matching in LAB space
    for channel in range(3):
        src_channel = source_lab[:, :, channel].flatten()
        ref_channel = reference_lab[:, :, channel].flatten()
        
        # Sort the arrays
        src_sorted = np.sort(src_channel)
        ref_sorted = np.sort(ref_channel)
        
        # Create mapping from source to reference distribution
        src_indices = np.searchsorted(src_sorted, src_channel)
        src_indices = np.clip(src_indices, 0, len(ref_sorted) - 1)
        
        # Map source values to reference distribution
        mapped_channel = ref_sorted[src_indices].reshape(source_lab.shape[:2])
        
        # Apply histogram mapping directly
        result_lab[:, :, channel] = mapped_channel
    
    # Apply Gaussian smoothing to reduce artifacts
    for channel in range(3):
        diff = result_lab[:, :, channel] - source_lab[:, :, channel]
        smooth_diff = ndimage.gaussian_filter(diff, sigma=sigma)
        result_lab[:, :, channel] = source_lab[:, :, channel] + smooth_diff
    
    # Convert back to RGB
    result = lab_to_rgb(result_lab)
    
    # Clip to valid RGB range
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)

def apply_fix_vae_color(current_image, reference_image, transfer_strength=1.0):
    """
    Apply LAB histogram color correction to current_image based on reference_image.
    """
    if reference_image is None:
        return current_image
    
    if PRINT_DEBUG_MESSAGES:
        print(f"[WhirlpoolUpscaler DEBUG] Color Correction: current shape {current_image.shape}, reference shape {reference_image.shape} [{get_debug_time()}ms]")
    
    batch_size, height, width, channels = current_image.shape
    ref_height, ref_width = reference_image.shape[1], reference_image.shape[2]
    
    # Resize reference image to match current image size for proper comparison
    if ref_height != height or ref_width != width:
        if PRINT_DEBUG_MESSAGES:
            print(f"[WhirlpoolUpscaler DEBUG] Color Correction: resizing reference from {ref_width}x{ref_height} to {width}x{height} [{get_debug_time()}ms]")
        reference_resized = nodes.ImageScale().upscale(reference_image, "lanczos", width, height, crop="disabled")[0]
    else:
        reference_resized = reference_image
    
    # Convert torch tensors to numpy arrays
    # Convert from [B, H, W, C] to [H, W, C] and scale to 0-255
    source_np = (current_image[0].cpu().numpy() * 255).astype(np.uint8)
    reference_np = (reference_resized[0].cpu().numpy() * 255).astype(np.uint8)
    
    # Apply LAB histogram color transfer
    corrected_np = lab_histogram_color_transfer(source_np, reference_np, transfer_strength=transfer_strength)
    
    # Convert back to torch tensor and normalize to [0, 1]
    corrected_tensor = torch.from_numpy(corrected_np.astype(np.float32) / 255.0).to(current_image.device)
    corrected_image = corrected_tensor.unsqueeze(0)  # Add batch dimension back
    
    # Clamp to valid range [0, 1]
    corrected_image = torch.clamp(corrected_image, 0.0, 1.0)
    
    if PRINT_DEBUG_MESSAGES:
        print(f"[WhirlpoolUpscaler DEBUG] Color Correction complete [{get_debug_time()}ms]")
    
    return corrected_image

def vae_decode_tiled(vae, samples, use_tile=True, decode_size=512, overlap=DEFAULT_OVERLAP):
    """VAE decode with tiling support"""
    if PRINT_DEBUG_MESSAGES:
        print(f"[WhirlpoolUpscaler DEBUG] VAE Decode: use_tile={use_tile}, decode_size={decode_size} [{get_debug_time()}ms]")
    
    if use_tile:
        decoder = nodes.VAEDecodeTiled()
        if 'overlap' in inspect.signature(decoder.decode).parameters:
            pixels = decoder.decode(vae, samples, decode_size, overlap=overlap)[0]
        else:
            pixels = decoder.decode(vae, samples, decode_size)[0]
    else:
        pixels = nodes.VAEDecode().decode(vae, samples)[0]
    
    if PRINT_DEBUG_MESSAGES:
        print(f"[WhirlpoolUpscaler DEBUG] VAE Decode complete: output shape {pixels.shape} [{get_debug_time()}ms]")
    
    return pixels

def vae_encode_tiled(vae, pixels, use_tile=True, decode_size=512, overlap=DEFAULT_OVERLAP):
    """VAE encode with tiling support"""
    if PRINT_DEBUG_MESSAGES:
        print(f"[WhirlpoolUpscaler DEBUG] VAE Encode: input shape {pixels.shape}, use_tile={use_tile}, decode_size={decode_size} [{get_debug_time()}ms]")
    
    if use_tile:
        encoder = nodes.VAEEncodeTiled()
        if 'overlap' in inspect.signature(encoder.encode).parameters:
            samples = encoder.encode(vae, pixels, decode_size, overlap=overlap)[0]
        else:
            samples = encoder.encode(vae, pixels, decode_size)[0]
    else:
        samples = nodes.VAEEncode().encode(vae, pixels)[0]
    
    if PRINT_DEBUG_MESSAGES:
        samples_tensor = extract_latent_tensor(samples)
        print(f"[WhirlpoolUpscaler DEBUG] VAE Encode complete: output shape {samples_tensor.shape} [{get_debug_time()}ms]")
    
    return samples

def latent_upscale_on_pixel_space(samples, resize_filter, w, h, vae, use_tile=True, decode_size=512, overlap=DEFAULT_OVERLAP, upscale_model=None):
    """Latent upscaling via pixel space conversion with optional AI upscale model"""
    # Ensure samples is in the proper format for VAE decode
    samples_dict = ensure_latent_dict(samples)
    
    # Step 1: VAE decode
    pixels = vae_decode_tiled(vae, samples_dict, use_tile, decode_size, overlap)
    
    # Step 2: Image upscale using wlsh_nodes approach - model upscale then resize to factor
    if upscale_model is not None:
        target_w = int(w)
        target_h = int(h)
        current_w = pixels.shape[2]
        current_h = pixels.shape[1]
        
        # Calculate the scale factor needed
        scale_factor = max(target_w / current_w, target_h / current_h)
        
        # Apply model upscaling once for quality
        pixels = upscale_with_model(upscale_model, pixels, decode_size)
        
        # Check if model actually upscaled (detect 1x models)
        new_w = pixels.shape[2]
        new_h = pixels.shape[1]
        if new_w == current_w and new_h == current_h:
            if PRINT_DEBUG_MESSAGES:
                print(f"[WhirlpoolUpscaler DEBUG] 1x upscale model detected, using standard scaling [{get_debug_time()}ms]")
            # Fall back to standard scaling
            pixels = nodes.ImageScale().upscale(pixels, resize_filter, target_w, target_h, False)[0]
        else:
            # Now resize the model-upscaled image to exact target dimensions
            # Use the scale factor relative to the ORIGINAL size, not the model-upscaled size
            final_w = int(current_w * scale_factor)
            final_h = int(current_h * scale_factor)
            
            if PRINT_DEBUG_MESSAGES:
                print(f"[WhirlpoolUpscaler DEBUG] Resizing model output from {new_w}x{new_h} to {final_w}x{final_h} (factor: {scale_factor:.3f}x) [{get_debug_time()}ms]")
            
            # Use ComfyUI's common_upscale for the final resize
            import comfy.utils
            samples = pixels.movedim(-1, 1)  # Convert to [B, C, H, W] for common_upscale
            s = comfy.utils.common_upscale(samples, final_w, final_h, resize_filter, crop="disabled")
            pixels = s.movedim(1, -1)  # Convert back to [B, H, W, C]
    else:
        # Use standard image scaling
        pixels = nodes.ImageScale().upscale(pixels, resize_filter, int(w), int(h), False)[0]
    
    # Step 3: VAE encode  
    upscaled_latent = vae_encode_tiled(vae, pixels, use_tile, decode_size, overlap)
    
    return upscaled_latent, pixels


@torch.no_grad()
def lanczos_upscale(image, resize_filter="lanczos", target_height=None, target_width=None, scale_factor=1.0):
    """Upscale image using ComfyUI's ImageScale node with proper Lanczos support."""
    if scale_factor == 1.0 and target_height is None and target_width is None:
        return image
    
    if PRINT_DEBUG_MESSAGES:
        print(f"[WhirlpoolUpscaler DEBUG] Image Resize: input shape {image.shape}, filter={resize_filter}, scale_factor={scale_factor} [{get_debug_time()}ms]")
    
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
    
    if PRINT_DEBUG_MESSAGES:
        print(f"[WhirlpoolUpscaler DEBUG] Image Resize: {width}x{height} -> {new_width}x{new_height} [{get_debug_time()}ms]")
    
    # Use ComfyUI's native ImageScale node for proper Lanczos upscaling
    upscaled = nodes.ImageScale().upscale(image, resize_filter, new_width, new_height, crop="disabled")[0]
    
    if PRINT_DEBUG_MESSAGES:
        print(f"[WhirlpoolUpscaler DEBUG] Image Resize complete: output shape {upscaled.shape} [{get_debug_time()}ms]")
    
    return upscaled


def common_upscaler(model, seed, steps_start, steps_end, cfg_start, cfg_end, sampler_name, scheduler, positive, negative, image, denoise_start=1.0, vae=None, iterations=4, denoise_end=0.05, upscale_by=1.0, resize_filter="lanczos", upscale_curve=1.0, decode_size=512, add_noise=0.0, fix_vae_color=False, upscale_model=None):
    if PRINT_DEBUG_MESSAGES:
        reset_debug_timer()  # Reset timer at start of process
        print(f"[WhirlpoolUpscaler DEBUG] Starting upscaling process: iterations={iterations}, upscale_by={upscale_by}x, steps={steps_start}->{steps_end}, cfg={cfg_start}->{cfg_end}, denoise={denoise_start}->{denoise_end} [{get_debug_time()}ms]")
    
    # Ensure image is in [B, H, W, C] format
    if len(image.shape) == 4 and image.shape[1] in [1, 3]:
        image = image.permute(0, 2, 3, 1)
    
    # If iterations is 0, return original image unchanged
    if iterations == 0:
        if PRINT_DEBUG_MESSAGES:
            print(f"[WhirlpoolUpscaler DEBUG] Zero iterations, returning original image [{get_debug_time()}ms]")
        return image
    
    # If VAE is not provided, use simple upscaling
    if vae is None:
        if PRINT_DEBUG_MESSAGES:
            print(f"[WhirlpoolUpscaler DEBUG] No VAE provided, using basic image upscaling [{get_debug_time()}ms]")
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
            # Reverse the curve to lean towards denoise_end instead of denoise_start
            curved_t = 1.0 - ((1.0 - t) ** upscale_curve)
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
            # Reverse the curve to lean towards steps_end instead of steps_start
            curved_t = 1.0 - ((1.0 - t) ** upscale_curve)
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
            # Reverse the curve to lean towards cfg_end instead of cfg_start
            curved_t = 1.0 - ((1.0 - t) ** upscale_curve)
            current_cfg = cfg_start + curved_t * (cfg_end - cfg_start)
            cfg_values.append(current_cfg)
    
    
    current_seed = seed
    
    # Encode original image to latent space once
    # This preserves the original image characteristics throughout iterations
    if PRINT_DEBUG_MESSAGES:
        print(f"[WhirlpoolUpscaler DEBUG] Encoding original image to latent space [{get_debug_time()}ms]")
    original_latent = vae_encode_tiled(vae, image, use_tile=True, decode_size=decode_size, overlap=DEFAULT_OVERLAP)
    current_latent = original_latent
    
    
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
            print(f"[WhirlpoolUpscaler] Iteration {iteration + 1}/{iterations} | Current: {current_width}x{current_height} | Target: {target_width}x{target_height} | Scale: {current_scale:.3f}x | Denoise: {current_denoise:.3f} | Steps: {current_steps} | CFG: {current_cfg:.1f} | Add Noise: {current_noise:.3f}")
        else:
            print(f"[WhirlpoolUpscaler] Iteration {iteration + 1}/{iterations} | Current: {current_width}x{current_height} | Target: {target_width}x{target_height} | Scale: {current_scale:.3f}x | Denoise: {current_denoise:.3f} | Steps: {current_steps} | CFG: {current_cfg:.1f}")
        
        try:
            # Upscale and sample method
            
            # Memory management
            model_management.soft_empty_cache()
            
            # STEP 1: Latent upscaling via pixel space
            upscaled_latent, upscaled_pixels = latent_upscale_on_pixel_space(
                current_latent, resize_filter, target_width, target_height, vae, 
                use_tile=True, decode_size=decode_size, overlap=DEFAULT_OVERLAP,
                upscale_model=upscale_model
            )
            
            # STEP 2: Apply color correction if enabled (after upscaling, before sampling) - skip on final iteration
            if fix_vae_color and iteration < iterations - 1:
                # Always use original image as reference (100% original image)
                reference_for_correction = image
                
                # Use 100% strength for all iterations (each iteration is a different image)
                transfer_strength = 1.0
                
                upscaled_pixels = apply_fix_vae_color(upscaled_pixels, reference_for_correction, transfer_strength)
                # Re-encode the color-corrected pixels
                upscaled_latent = vae_encode_tiled(vae, upscaled_pixels, use_tile=True, decode_size=decode_size, overlap=DEFAULT_OVERLAP)
            
            # STEP 3: Sampling
            
            latent_tensor = extract_latent_tensor(upscaled_latent)
            
            # Add noise before denoising if noise level > 0
            if current_noise > 0.0:
                if PRINT_DEBUG_MESSAGES:
                    print(f"[WhirlpoolUpscaler DEBUG] Adding noise: {current_noise:.3f} [{get_debug_time()}ms]")
                # Generate noise tensor with same shape as latent
                additional_noise = torch.randn_like(latent_tensor) * current_noise
                latent_tensor = latent_tensor + additional_noise
            
            if PRINT_DEBUG_MESSAGES:
                print(f"[WhirlpoolUpscaler DEBUG] Sampling: {current_steps} steps, {current_cfg} CFG, {current_denoise:.3f} denoise, sampler={sampler_name} [{get_debug_time()}ms]")
            
            noise = comfy.sample.prepare_noise(latent_tensor, current_seed, None)
            
            refined_samples = comfy.sample.sample(
                model, noise, current_steps, current_cfg, sampler_name, scheduler, positive, negative, latent_tensor,
                denoise=current_denoise, disable_noise=False, start_step=None, last_step=None,
                force_full_denoise=True, noise_mask=None, seed=current_seed
            )
            refined_latent = ensure_latent_dict(refined_samples)
            
            if PRINT_DEBUG_MESSAGES:
                print(f"[WhirlpoolUpscaler DEBUG] Sampling complete for iteration {iteration + 1} [{get_debug_time()}ms]")
            
            # STEP 4: Update current latent for next iteration
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
                    fallback_image = vae_decode_tiled(vae, current_latent, use_tile=True, decode_size=decode_size)
                    fallback_upscaled = lanczos_upscale(fallback_image, resize_filter, target_height=target_height, target_width=target_width)
                    # Re-encode for consistency
                    current_latent = vae_encode_tiled(vae, fallback_upscaled, use_tile=True, decode_size=decode_size)
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
        

    # Final decode fallback (only reached if final iteration decode failed)
    if PRINT_DEBUG_MESSAGES:
        print(f"[WhirlpoolUpscaler DEBUG] Final decode fallback: converting latent to image [{get_debug_time()}ms]")
    try:
        final_image = vae_decode_tiled(vae, current_latent, use_tile=True, decode_size=decode_size, overlap=DEFAULT_OVERLAP)
        
        if PRINT_DEBUG_MESSAGES:
            print(f"[WhirlpoolUpscaler DEBUG] Upscaling process complete: final shape {final_image.shape} [{get_debug_time()}ms]")
        return final_image
    except Exception as e:
        print(f"Final decode failed: {e}, using fallback decode")
        try:
            # Fallback to simpler decode
            final_image = vae_decode_tiled(vae, current_latent, use_tile=False)
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
                "upscale_curve": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Progression curve for all parameters. 1.0 = linear. >1.0: resolution accelerates, CFG/steps/denoise front-loaded. <1.0: resolution front-loaded, CFG/steps/denoise accelerate."}),
                "iterations": ("INT", {"default": 4, "min": 0, "max": 20, "tooltip": "Number of complete sampling cycles to perform."}),
                "steps_start": ("INT", {"default": 17, "min": 1, "max": 10000, "tooltip": "Number of sampling steps for the first iteration."}),
                "steps_end": ("INT", {"default": 7, "min": 1, "max": 10000, "tooltip": "Number of sampling steps for the last iteration."}),
                "cfg_start": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01, "tooltip": "CFG scale for the first iteration."}),
                "cfg_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01, "tooltip": "CFG scale for the last iteration."}),
                "denoise_start": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise strength for the first iteration."}),
                "denoise_end": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise strength for the last iteration."}),
                "add_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Adds noise before denoising each iteration. The amount added is relative to the current denoise."}),
                "fix_vae_color": ("BOOLEAN", {"default": True, "tooltip": "Apply color correction after each iteration to maintain color consistency with the original image."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_2m"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "bong_tangent"}),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "resize_filter": (["lanczos", "nearest-exact", "bilinear", "area", "bicubic"], {"default": "lanczos", "tooltip": "Image resizing filter algorithm."}),
                "decode_size": ("INT", {"default": 1024, "min": 320, "max": 2048, "step": 64, "tooltip": "Decode size for VAE operations."}),
                "vae": ("VAE", ),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Optional AI upscaling model (e.g., ESRGAN, Real-ESRGAN) for enhanced image quality during upscaling."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"

    CATEGORY = "Whirlpool Upscaler"

    def sample(self, model, seed, iterations, steps_start, steps_end, cfg_start, cfg_end, sampler_name, scheduler, positive, negative, image, denoise_start=1.0, denoise_end=1.0, add_noise=0.0, fix_vae_color=False, upscale_by=1.0, resize_filter="lanczos", upscale_curve=1.0, decode_size=512, vae=None, upscale_model=None):
        image_out = common_upscaler(model, seed, steps_start, steps_end, cfg_start, cfg_end, sampler_name, scheduler, positive, negative, image, denoise_start=denoise_start, vae=vae, iterations=iterations, denoise_end=denoise_end, upscale_by=upscale_by, resize_filter=resize_filter, upscale_curve=upscale_curve, decode_size=decode_size, add_noise=add_noise, fix_vae_color=fix_vae_color, upscale_model=upscale_model)
        
        if image_out is None:
            # Create a black image if everything fails
            # Get the shape from the input image and rescale factor
            height = int(image.shape[1] * upscale_by)
            width = int(image.shape[2] * upscale_by)
            image_out = torch.zeros((image.shape[0], height, width, 3), dtype=torch.float32, device="cpu")

        return (image_out,) 
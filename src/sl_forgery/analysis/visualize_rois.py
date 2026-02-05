import tifffile
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import zoom
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union


# TODO we would like this to be callable from the command line, plus functionality in the batch processing function.
# Core Processing Functions

def load_tiff_data(
        tiff_path: Union[str, Path],
        max_frames: Optional[int] = None,
        verbose: bool = False
) -> np.ndarray:
    """Load TIFF data from file.

    Args:
        tiff_path: Path to the TIFF file.
        max_frames: Maximum number of frames to load. If None, loads all frames.
        verbose: Whether to print loading progress.

    Returns:
        Array of shape (num_frames, height, width) containing the TIFF data.

    Raises:
        FileNotFoundError: If tiff_path doesn't exist.
    """
    tiff_path = Path(tiff_path)

    if not tiff_path.exists():
        raise FileNotFoundError(f"TIFF file not found: {tiff_path}")

    with tifffile.TiffFile(tiff_path) as tif:
        total_pages = len(tif.pages)
        num_frames = min(max_frames, total_pages) if max_frames else total_pages

        if verbose:
            print(f"Found {total_pages} frames, loading {num_frames}...")

        frames = []
        for i in range(num_frames):
            frame = tif.pages[i].asarray()
            frames.append(frame)
            if verbose and (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{num_frames} frames")

        raw_data = np.array(frames)

    if verbose:
        print(f"Loaded data: {raw_data.shape}")

    return raw_data


def extract_roi_info(
        roi_metadata: List[Dict],
        verbose: bool = False
) -> List[Dict]:
    """Extract ROI information from metadata.

    Args:
        roi_metadata: List of ROI metadata dictionaries from ScanImage.
        verbose: Whether to print ROI information.

    Returns:
        List of dictionaries containing extracted ROI information with keys:
            - name: ROI name
            - size: [width, height] in pixels
            - center: [x, y] in microns
            - physical_size: [width, height] in microns
            - pixel_scale_x: microns per pixel in X
            - pixel_scale_y: microns per pixel in Y
    """
    rois = []

    for roi in roi_metadata:
        sf = roi['scanfields']

        # Calculate actual pixel scale (µm per pixel)
        pixel_scale_x = sf['sizeXY'][0] / sf['pixelResolutionXY'][0]
        pixel_scale_y = sf['sizeXY'][1] / sf['pixelResolutionXY'][1]

        roi_info = {
            'name': roi['name'],
            'size': sf['pixelResolutionXY'],  # [width, height] in pixels
            'center': sf['centerXY'],  # [x, y] in microns
            'physical_size': sf['sizeXY'],  # [width, height] in microns
            'pixel_scale_x': pixel_scale_x,
            'pixel_scale_y': pixel_scale_y
        }
        rois.append(roi_info)

        if verbose:
            print(f"{roi_info['name']}:")
            print(f"  Size: {roi_info['size'][0]}x{roi_info['size'][1]} pixels")
            print(f"  Physical size: {roi_info['physical_size'][0]:.3f}x{roi_info['physical_size'][1]:.3f} µm")
            print(f"  Pixel scale X: {pixel_scale_x:.6f} um/pixel")
            print(f"  Pixel scale Y: {pixel_scale_y:.6f} um/pixel")

    return rois


def calculate_zoom_factors(
        rois: List[Dict],
        verbose: bool = False
) -> Tuple[float, float, float]:
    """Calculate zoom factors to make square pixels (otherwise they are long rectangles and the image is squished)

    Uses the first ROI as reference. The pixel ratio during imaging was 0.75:0.5 (X:Y).

    Args:
        rois: List of ROI info dictionaries from extract_roi_info().
        verbose: Whether to print zoom factor information.

    Returns:
        Tuple of (zoom_x, zoom_y, final_pixel_scale) where:
            - zoom_x: Zoom factor for X dimension (typically 1.0)
            - zoom_y: Zoom factor for Y dimension (typically ~1.5)
            - final_pixel_scale: Final pixel scale in microns per pixel
    """
    ref_roi = rois[0]
    zoom_x = 1.0
    zoom_y = ref_roi['pixel_scale_y'] / ref_roi['pixel_scale_x']
    final_pixel_scale = ref_roi['pixel_scale_x']

    if verbose:
        print(f"\nZoom factors: X={zoom_x:.3f}, Y={zoom_y:.3f}")
        print(f"Final pixel scale: {final_pixel_scale:.6f} µm/pixel")

    return zoom_x, zoom_y, final_pixel_scale


def extract_and_resample_rois(
        raw_data: np.ndarray,
        rois: List[Dict],
        zoom_x: float,
        zoom_y: float,
        flyback_pixels: int = 122,
        verbose: bool = False
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Extract ROIs from vertical stack and resample to square pixels.

    The ROIs are stacked vertically in the raw data with flyback regions between them.
    This function extracts each ROI, detects if trimming was applied, updates metadata
    accordingly, and resamples to square pixels.

    Args:
        raw_data: Array of shape (num_frames, height, width).
        rois: List of ROI info dictionaries (will be modified in place).
        zoom_x: Zoom factor for X dimension.
        zoom_y: Zoom factor for Y dimension.
        flyback_pixels: Number of flyback pixels between ROIs.
        verbose: Whether to print processing information.

    Returns:
        Tuple of (extracted_rois, updated_rois) where:
            - extracted_rois: List of arrays, each with shape (num_frames, height, width)
            - updated_rois: Updated ROI info dictionaries with detected trimming applied
    """
    num_frames = raw_data.shape[0]
    current_row = 0
    extracted_rois = []

    for roi_idx, roi in enumerate(rois):
        height = roi['size'][1]  # Height in pixels from metadata
        roi_data = []

        for frame_idx in range(num_frames):
            frame = raw_data[frame_idx]
            roi_frame = frame[current_row:current_row + height, :]

            # Check if trimming was applied by comparing metadata width with actual width
            roi_meta_width = roi['size'][0]  # Expected width in pixels from metadata
            roi_actual_width = roi_frame.shape[1]  # Actual width in pixels

            if roi_actual_width != roi_meta_width:
                pixels_removed = roi_meta_width - roi_actual_width

                # Update metadata only once (on first frame)
                if frame_idx == 0:
                    physical_width_removed = pixels_removed * roi['pixel_scale_x']

                    # Shift centers to account for removed pixels
                    # ROI 1 (index 0) stays as reference
                    # ROI 2 (index 1) shifts left by 1x the trimmed amount
                    # ROI 3 (index 2) shifts left by 2x the trimmed amount
                    if roi_idx > 0:
                        center_shift = roi_idx * physical_width_removed
                        roi['center'][0] -= center_shift

                    # Update pixel size
                    roi['size'][0] = roi_actual_width

                    # Update physical size
                    roi['physical_size'][0] -= physical_width_removed

                    if verbose:
                        print(f"{roi['name']}: Detected {pixels_removed} pixels "
                              f"({physical_width_removed:.3f} µm) trimmed from left edge")
                        if roi_idx > 0:
                            print(f"  Center shifted LEFT by {center_shift:.3f} µm")

            # Resample to square pixels using scipy zoom (linear interpolation)
            roi_frame = zoom(roi_frame, [zoom_y, zoom_x], order=1)
            roi_data.append(roi_frame)

        extracted_rois.append(np.array(roi_data))
        current_row += height + flyback_pixels

        # Update ROI info with resampled size
        roi['resampled_size'] = extracted_rois[roi_idx][0].shape[::-1]  # (width, height)

        if verbose:
            print(f"{roi['name']}: {roi['size'][0]}x{roi['size'][1]} → "
                  f"{roi['resampled_size'][0]}x{roi['resampled_size'][1]} pixels")

    return extracted_rois, rois


def calculate_alignment(
        rois: List[Dict],
        final_pixel_scale: float,
        verbose: bool = False
) -> List[Dict]:
    """Calculate spatial alignment for ROIs.

    Converts relative ROI locations from microns (in metadata) to pixels for alignment.
    Uses the first ROI as the reference point (0, 0).

    Args:
        rois: List of ROI info dictionaries (will be modified in place).
        final_pixel_scale: Final pixel scale in microns per pixel.
        verbose: Whether to print alignment information.

    Returns:
        Updated ROI info dictionaries with 'offset_pixels' added to each.
    """
    ref_center = rois[0]['center']  # Use first ROI as reference

    if verbose:
        print("\nROI alignment:")

    for roi in rois:
        # Calculate offset in microns
        offset_microns = [
            roi['center'][0] - ref_center[0],
            roi['center'][1] - ref_center[1]
        ]

        # Convert to pixels using final square pixel scale
        roi['offset_pixels'] = [
            offset_microns[0] / final_pixel_scale,
            offset_microns[1] / final_pixel_scale
        ]

        if verbose:
            print(f"  {roi['name']}: offset X={roi['offset_pixels'][0]:.1f}, "
                  f"Y={roi['offset_pixels'][1]:.1f} pixels")

    return rois


def create_composites(
        extracted_rois: List[np.ndarray],
        rois: List[Dict],
        verbose: bool = False
) -> np.ndarray:
    """Create composite images with aligned ROIs.

    Places each ROI at its correct spatial position on a canvas, creating a single
    aligned image from multiple ROIs.

    Args:
        extracted_rois: List of arrays, each with shape (num_frames, height, width).
        rois: List of ROI info dictionaries with alignment information.
        verbose: Whether to print canvas information.

    Returns:
        Array of shape (num_frames, canvas_height, canvas_width) containing
        the aligned composite images.
    """
    # Calculate bounding box using actual resampled sizes
    min_x = min(roi['offset_pixels'][0] - roi['resampled_size'][0] / 2 for roi in rois)
    max_x = max(roi['offset_pixels'][0] + roi['resampled_size'][0] / 2 for roi in rois)
    min_y = min(roi['offset_pixels'][1] - roi['resampled_size'][1] / 2 for roi in rois)
    max_y = max(roi['offset_pixels'][1] + roi['resampled_size'][1] / 2 for roi in rois)

    canvas_width = int(np.ceil(max_x - min_x))
    canvas_height = int(np.ceil(max_y - min_y))

    if verbose:
        print(f"\nCanvas size: {canvas_width}x{canvas_height} pixels")

    # Calculate canvas positions for each ROI
    for roi in rois:
        roi['canvas_pos'] = [
            int(roi['offset_pixels'][0] - min_x - roi['resampled_size'][0] / 2),
            int(roi['offset_pixels'][1] - min_y - roi['resampled_size'][1] / 2)
        ]

        if verbose:
            print(f"  {roi['name']}: canvas position X={roi['canvas_pos'][0]}, "
                  f"Y={roi['canvas_pos'][1]}")

    num_frames = len(extracted_rois[0])
    composites = []

    # Create composite for each frame
    for frame_idx in range(num_frames):
        composite = np.zeros((canvas_height, canvas_width), dtype=extracted_rois[0].dtype)

        for roi_idx, roi in enumerate(rois):
            img = extracted_rois[roi_idx][frame_idx]
            x, y = roi['canvas_pos']
            h, w = img.shape
            composite[y:y + h, x:x + w] = img

        composites.append(composite)

    return np.array(composites)


def process_single_tiff(
        tiff_path: Union[str, Path],
        roi_metadata: List[Dict],
        flyback_pixels: int = 122,
        max_frames: Optional[int] = None,
        verbose: bool = False
) -> np.ndarray:
    """Process a single TIFF file and return aligned composites.

    This is the main processing pipeline that combines all the steps:
    loading, ROI extraction, resampling, alignment, and composite creation.

    Args:
        tiff_path: Path to the TIFF file.
        roi_metadata: List of ROI metadata dictionaries.
        flyback_pixels: Number of flyback pixels between ROIs.
        max_frames: Maximum number of frames to process. If None, processes all.
        verbose: Whether to print processing information.

    Returns:
        Array of shape (num_frames, height, width) containing aligned composites.
    """
    # Load data
    raw_data = load_tiff_data(tiff_path, max_frames=max_frames, verbose=verbose)

    # Extract ROI info
    rois = extract_roi_info(roi_metadata, verbose=verbose)

    # Calculate zoom factors
    zoom_x, zoom_y, final_pixel_scale = calculate_zoom_factors(rois, verbose=verbose)

    # Extract and resample ROIs
    extracted_rois, rois = extract_and_resample_rois(
        raw_data, rois, zoom_x, zoom_y, flyback_pixels, verbose=verbose
    )

    # Calculate alignment
    rois = calculate_alignment(rois, final_pixel_scale, verbose=verbose)

    # Create composites
    composites = create_composites(extracted_rois, rois, verbose=verbose)

    if verbose:
        print(f"\nProcessing complete!")
        print(f"Created {len(composites)} composite frames")
        print(f"Composite size: {composites[0].shape}")

    return composites


# Batch Processing Functions

# TODO this needs a way to check if the files in the output folder have already been processed (they share the same
#  filename as the original Tiffs, just with "_trimmed" added),
#  to avoid reprocessing, and also it would be great if there was an option to NOT save the files but just generate
#  them here for the visualization (memory-mapped). AND it would be nice if there was an option for random selection of
#  files, and you said give me 10 "random" files and it processed evenly spaced files (like 0, 14, 28,
#  etc). In general this should avoid the last file as it usually only has a handful of frames
def batch_process(
        folder_path: Union[str, Path],
        metadata: List[Dict],
        output_dir: Union[str, Path],
        save_format: str = 'npz',
        convert_to_uint8: bool = True,
        max_files: Optional[int] = None,
        flyback_pixels: int = 122,
        preview_first: bool = True,
        verbose: bool = False
) -> None:
    """Process all TIFF files in a folder and save as compressed NPZ or TIFF (or not save them at all and generate
    dynamically, needs this added)

    Args:
        folder_path: Directory containing TIFF files to process.
        metadata: List of ROI metadata dictionaries (same for all files).
        output_dir: Directory to save processed files.
        save_format: Format to save files. Either 'npz' (compressed, ~600MB per file)
            or 'tiff' (uncompressed, ~1.5GB per file).
        convert_to_uint8: If True, normalize to uint8 for much better compression (takes longer)
        max_files: Maximum number of files to process. If None, processes all.
        flyback_pixels: Number of flyback pixels between ROIs.
        preview_first: Whether to show a preview and ask for confirmation before processing.
        verbose: Whether to print detailed processing information.

    Raises:
        ValueError: If no TIFF files found or if save_format is invalid.
    """
    folder_path = Path(folder_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all TIFF files
    tiff_files = sorted(set(
        list(folder_path.glob('*.tif')) + list(folder_path.glob('*.tiff'))
    ))

    if not tiff_files:
        raise ValueError(f"No TIFF files found in {folder_path}")

    # Limit number of files if specified
    if max_files is not None:
        tiff_files = tiff_files[:max_files]

    if save_format not in ['npz', 'tiff']:
        raise ValueError(f"save_format must be 'npz' or 'tiff', got '{save_format}'")

    print(f"\n{'=' * 60}")
    print(f"Batch processing: {len(tiff_files)} files found")
    print(f"Save format: {save_format}")
    print(f"Convert to uint8: {convert_to_uint8}")
    print(f"{'=' * 60}\n")

    # Preview first file (if requested) to see if it looks right before moving on
    if preview_first:
        print(f"Processing first file for preview: {tiff_files[0].name}")
        preview_composites = process_single_tiff(
            tiff_path=tiff_files[0],
            roi_metadata=copy.deepcopy(metadata),
            flyback_pixels=flyback_pixels,
            max_frames=100,
            verbose=False
        )

        # Show preview
        first_frame = preview_composites[0]
        vmin = np.percentile(first_frame, 1)
        vmax = np.percentile(first_frame, 99.5)

        plt.figure(figsize=(10, 8))
        plt.imshow(first_frame, cmap='gray', vmin=vmin, vmax=vmax)
        plt.title(f'Preview: First frame of {tiff_files[0].name}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        response = input(f"\nContinue processing all {len(tiff_files)} files? [y/n]: ")
        if response.lower() != 'y':
            print("Batch processing cancelled.")
            return

    # Process all files
    for idx, tiff_file in enumerate(tiff_files, 1):
        print(f"\n[{idx}/{len(tiff_files)}] Processing: {tiff_file.name}")

        # Use deepcopy to avoid modifying original metadata
        file_metadata = copy.deepcopy(metadata)

        composites = process_single_tiff(
            tiff_path=tiff_file,
            roi_metadata=file_metadata,
            flyback_pixels=flyback_pixels,
            verbose=verbose
        )

        # Convert to uint8 if requested (for better compression)
        if convert_to_uint8:
            # Calculate percentile-based normalization to preserve contrast
            vmin = np.percentile(composites, 1)
            vmax = np.percentile(composites, 99.5)

            if verbose:
                print(f"  Converting to uint8 (range: {vmin:.1f} to {vmax:.1f})")

            # Normalize to 0-255
            composites_normalized = np.clip(
                (composites - vmin) / (vmax - vmin) * 255, 0, 255
            ).astype(np.uint8)

            composites = composites_normalized

        # Check data type and size before saving
        if verbose:
            print(f"  Composite dtype: {composites.dtype}")
            print(f"  Composite shape: {composites.shape}")
            uncompressed_size = composites.nbytes / (1024 * 1024)
            print(f"  Uncompressed size: {uncompressed_size:.1f} MB")

        # Save in specified format
        if save_format == 'npz':
            output_path = output_dir / f"{tiff_file.stem}_aligned.npz"

            # Save with compression
            np.savez_compressed(output_path, composites=composites)

            file_size_mb = output_path.stat().st_size / (1024 * 1024)

            print(f"  Saved compressed NPZ: {output_path}")
            print(f"  File size: {file_size_mb:.1f} MB", end="")
            if verbose:
                compression_ratio = uncompressed_size / file_size_mb
                print(f" (compression ratio: {compression_ratio:.1f}x)", end="")
            print()

        elif save_format == 'tiff':
            output_path = output_dir / f"{tiff_file.stem}_aligned.tif"
            tifffile.imwrite(output_path, composites, compression='zlib',
                             compressionargs={'level': 6})
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  Saved compressed TIFF: {output_path} ({file_size_mb:.1f} MB)")

        print(f"  Processed {len(composites)} frames")

    print(f"\n{'=' * 60}")
    print(f"Batch processing complete!")
    print(f"Saved {len(tiff_files)} files to {output_dir}")
    print(f"{'=' * 60}\n")


def load_composites_from_npz(
        input_folder: Union[str, Path],
        max_files: Optional[int] = None,
        verbose: bool = False
) -> np.ndarray:
    """Load composite frames from saved NPZ files.

    Args:
        input_folder: Directory containing NPZ files.
        max_files: Maximum number of files to load. If None, loads all.
        verbose: Whether to print loading information.

    Returns:
        Array of shape (total_frames, height, width) containing all composites.

    Raises:
        ValueError: If no NPZ files found in input_folder.
    """
    input_folder = Path(input_folder)

    npz_files = sorted(input_folder.glob('*_aligned.npz'))

    if not npz_files:
        raise ValueError(f"No NPZ files found in {input_folder}")

    # Limit number of files if specified
    if max_files is not None:
        npz_files = npz_files[:max_files]

    print(f"\n{'=' * 60}")
    print(f"Loading composites from {len(npz_files)} NPZ files")
    print(f"{'=' * 60}\n")

    all_composites = []

    for idx, npz_file in enumerate(npz_files, 1):
        if verbose:
            print(f"[{idx}/{len(npz_files)}] Loading: {npz_file.name}")
        data = np.load(npz_file)
        composites = data['composites']
        all_composites.append(composites)
        if verbose:
            print(f"  Loaded {len(composites)} frames (total so far: {sum(len(c) for c in all_composites)})")

    # Concatenate all composites
    all_composites = np.concatenate(all_composites, axis=0)

    print(f"\n{'=' * 60}")
    print(f"Loading complete! Total frames: {len(all_composites)}")
    print(f"Frame shape: {all_composites[0].shape}")
    print(f"{'=' * 60}\n")

    return all_composites


# visualization Functions (matplotlib is best but napari is another option, but much slower)
# TODO also this creates a window that on my mac is bigger than the screen but I can resize it.  Potential fix to
#  open as fullscreen? Idk
def visualize_composites_interactive(
        composites: np.ndarray,
        vmin_percentile: float = 1.0,
        vmax_percentile: float = 99.5,
        figsize: Tuple[int, int] = (14, 10)
) -> Tuple[plt.Figure, plt.Axes, Slider]:
    """Create interactive matplotlib visualization with slider for composite frames.

    Args:
        composites: Array of shape (num_frames, height, width).
        vmin_percentile: Lower percentile for contrast adjustment.
        vmax_percentile: Upper percentile for contrast adjustment.
        figsize: Figure size as (width, height) in inches.

    Returns:
        Tuple of (figure, axes, slider) for the interactive plot.

    Raises:
        ValueError: If composites array is empty.
    """
    if len(composites) == 0:
        raise ValueError("No composite frames provided")

    # Calculate contrast limits from first frame
    vmin = np.percentile(composites[0], vmin_percentile)
    vmax = np.percentile(composites[0], vmax_percentile)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.15)

    # Display first frame
    im = ax.imshow(composites[0], cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f'Frame 0 / {len(composites) - 1}')
    ax.axis('off')

    # Create slider
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, len(composites) - 1,
                    valinit=0, valstep=1)

    # Update function for slider
    def update(val: float) -> None:
        frame_idx = int(slider.val)
        im.set_data(composites[frame_idx])
        ax.set_title(f'Frame {frame_idx} / {len(composites) - 1}')
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

    return fig, ax, slider


# def visualize_with_napari(composites: np.ndarray) -> None:
#     """Visualize composites with napari viewer (better scrubbing than matplotlib).
#
#     Napari is a scientific image viewer with excellent performance for large datasets.
#     Install with: pip install napari[all]
#
#     Args:
#         composites: Array of shape (num_frames, height, width).
#     """
#     try:
#         import napari
#
#         viewer = napari.Viewer()
#         viewer.add_image(
#             composites,
#             name='Aligned Composites',
#             colormap='gray',
#             contrast_limits=[np.percentile(composites, 1), np.percentile(composites, 99.5)]
#         )
#
#         print("\nNapari viewer opened!")
#         print("Controls:")
#         print("  - Drag slider at bottom to scrub through frames")
#         print("  - Mouse wheel to zoom")
#         print("  - Click and drag to pan")
#         print("  - Home/End keys to jump to first/last frame\n")
#
#         napari.run()
#
#     except ImportError:
#         print("Napari not installed. Install with: pip install napari[all]")
#         print("Falling back to matplotlib viewer...")
#         visualize_composites_interactive(composites)


# Metadata Loading Functions

def load_metadata_from_json(json_path: Union[str, Path]) -> List[Dict]:
    """Load ROI metadata from ScanImage JSON file.

    Extracts ROI information from the nested JSON structure produced by ScanImage.
    The relevant data is in: RoiGroups > imagingRoiGroup > rois

    Args:
        json_path: Path to the JSON metadata file.

    Returns:
        List of dictionaries, each containing:
            - name: ROI name
            - scanfields: Dictionary with centerXY, sizeXY, and pixelResolutionXY

    Raises:
        FileNotFoundError: If json_path doesn't exist.
        ValueError: If JSON structure is invalid or missing required fields.
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, 'r') as f:
        metadata = json.load(f)

    # Handle different JSON structures
    if isinstance(metadata, list):
        # Already in correct format
        return metadata
    elif 'RoiGroups' in metadata:
        try:
            roi_groups = metadata['RoiGroups']
            imaging_group = roi_groups.get('imagingRoiGroup', {})
            rois = imaging_group.get('rois', [])

            if not rois:
                raise ValueError("No ROIs found in imagingRoiGroup")

            # Extract relevant fields for each ROI
            extracted_rois = []
            for roi in rois:
                roi_data = {
                    'name': roi.get('name', 'Unnamed ROI'),
                    'scanfields': {
                        'centerXY': roi['scanfields']['centerXY'],
                        'sizeXY': roi['scanfields']['sizeXY'],
                        'pixelResolutionXY': roi['scanfields']['pixelResolutionXY']
                    }
                }
                extracted_rois.append(roi_data)

            print(f"Loaded {len(extracted_rois)} ROIs from ScanImage JSON")
            return extracted_rois

        except KeyError as e:
            raise ValueError(f"Invalid ScanImage JSON structure: missing key {e}")
    else:
        raise ValueError(
            "Unknown JSON format. Expected either a list of ROIs or "
            "a dictionary with 'RoiGroups' key."
        )


# Main Execution
if __name__ == "__main__":
    # Load metadata
    metadata = load_metadata_from_json(
        '/Users/cs963/Desktop/sun_lab_projects/untitled folder/26_finalday_mesodata/frame_invariant_metadata.json'
    )

    batch_process(
                folder_path='/Users/cs963/Desktop/sun_lab_projects/untitled '
                            'folder/26_finalday_mesodata/26_lastday_trimmed',
                metadata=metadata,
                output_dir='/Users/cs963/Desktop/sun_lab_projects/untitled '
                           'folder/26_finalday_mesodata/26_lastday_aligned_trimmed_npz',
                save_format='npz',
                convert_to_uint8=True,
                max_files=2,
                flyback_pixels=122,
                preview_first=True,
                verbose=True        #again set this to flase to avoid seeing all the output
            )

    # Load composites (fast since already processed)
    composites = load_composites_from_npz(
        input_folder='/Users/cs963/Desktop/sun_lab_projects/untitled '
                     'folder/26_finalday_mesodata/26_lastday_aligned_trimmed_npz',
        max_files=2,
        verbose=True
    )

    # Interactive matplotlib viewer
    print("\n" + "=" * 60)
    print("Opening interactive viewer for scrubbing")
    print("=" * 60)

    visualize_composites_interactive(composites, figsize=(16, 12))

    #visualize_with_napari(composites)

# TODO it;s also important to make composites of the original corrupted files and visualize those to make sure A)
#  they need to be cropped and B) that the cropping is working

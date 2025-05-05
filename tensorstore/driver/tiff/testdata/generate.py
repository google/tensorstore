import numpy as np
import tifffile
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

OUTPUT_DIR = Path("raw")
BASE_HEIGHT = 32
BASE_WIDTH = 48
TILE_SHAPE = (16, 16)


def generate_coordinate_array(shape, dtype=np.uint16):
    shape = tuple(shape)
    arr = np.zeros(shape, dtype=dtype)
    it = np.nditer(arr, flags=["multi_index"], op_flags=["readwrite"])
    count = 1
    while not it.finished:
        if np.issubdtype(dtype, np.integer):
            iinfo = np.iinfo(dtype)
            modulo_base = int(iinfo.max) + 1
            if modulo_base > 0:
                current_val = count % modulo_base
            else:
                current_val = count
        else:
            current_val = count

        arr[it.multi_index] = current_val
        count += 1
        it.iternext()
    return arr


def write_tiff(
    filename: Path,
    base_shape: tuple,
    dtype: np.dtype,
    stack_dims: dict | None = None,
    spp: int = 1,
    planar_config_str: str = "contig",
    tile_shape: tuple | None = TILE_SHAPE,
    ifd_sequence_order: list[str] | None = None,
    photometric: str | None = None,
    extrasamples: tuple | None = None,
    compression: str | None = None,
    description: str | None = None,
):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Generating TIFF: {filename.name}")
    logging.info(
        f"  Stack: {stack_dims or 'None'}, SPP: {spp}, Planar: {planar_config_str}, Dtype: {dtype.__name__}, Tile: {tile_shape}"
    )

    stack_dims = stack_dims or {}

    if not stack_dims:
        stack_labels_numpy_order = []
        stack_shape_numpy_order = []
    elif ifd_sequence_order:
        stack_labels_numpy_order = ifd_sequence_order  # Slowest -> Fastest
        stack_shape_numpy_order = [
            stack_dims[label] for label in stack_labels_numpy_order
        ]
    else:
        # Default order: alphabetical for consistency if not specified
        stack_labels_numpy_order = sorted(stack_dims.keys())
        stack_shape_numpy_order = [
            stack_dims[label] for label in stack_labels_numpy_order
        ]
        logging.warning(
            f"  ifd_sequence_order not specified for {filename.name}, using default alphabetical order: {stack_labels_numpy_order}"
        )

    numpy_shape_list = list(stack_shape_numpy_order)
    height, width = base_shape

    if spp > 1 and planar_config_str == "separate":
        numpy_shape_list.append(spp)

    numpy_shape_list.extend([height, width])

    if spp > 1 and planar_config_str == "contig":
        numpy_shape_list.append(spp)

    full_shape = tuple(numpy_shape_list)
    logging.info(f"  Generating numpy data with shape: {full_shape}")

    full_data = generate_coordinate_array(full_shape, dtype=dtype)

    # Reshape for IFD slicing
    num_ifds = np.prod(stack_shape_numpy_order or [1])
    flat_ifd_data = (
        full_data.reshape((num_ifds, height, width, spp))
        if spp > 1 and planar_config_str == "contig"
        else full_data.reshape((num_ifds, height, width))
    )

    tifffile_kwargs = {
        "planarconfig": planar_config_str,
        "dtype": dtype,
        "shape": (
            (height, width, spp)
            if spp > 1 and planar_config_str == "contig"
            else (height, width)
        ),
    }

    if photometric:
        tifffile_kwargs["photometric"] = photometric
    if extrasamples:
        tifffile_kwargs["extrasamples"] = extrasamples
    if tile_shape:
        tifffile_kwargs["tile"] = tile_shape
    if compression:
        tifffile_kwargs["compression"] = compression
    if description:
        tifffile_kwargs["description"] = description

    try:
        for i in range(num_ifds):
            tifffile.imwrite(
                filename,
                flat_ifd_data[i],
                append=i > 0,
                **tifffile_kwargs,
            )
        logging.info(f"  Successfully wrote {filename.name}")
    except Exception as e:
        logging.error(f"  Failed to write {filename.name}: {e}")
        if filename.exists():
            os.remove(filename)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Starting TIFF file generation in {OUTPUT_DIR}")
    logging.info(
        f"Using Base Shape: {BASE_HEIGHT}x{BASE_WIDTH}, Tile Shape: {TILE_SHAPE}"
    )


# --- Test Case 1: Simple Z-Stack (5 planes), SPP=1 ---
write_tiff(
    filename=OUTPUT_DIR / "stack_z5_spp1_uint8.tif",
    base_shape=(BASE_HEIGHT, BASE_WIDTH),
    dtype=np.uint8,
    stack_dims={"z": 5},
    description="Z=5, SPP=1, uint8, Contig, Tile=16x16",
)

# --- Test Case 2: Z-Stack (4 planes), SPP=3 (RGB), Contig ---
write_tiff(
    filename=OUTPUT_DIR / "stack_z4_spp3_rgb_uint16.tif",
    base_shape=(BASE_HEIGHT, BASE_WIDTH),
    dtype=np.uint16,
    stack_dims={"z": 4},
    spp=3,
    planar_config_str="contig",
    photometric="rgb",  # Explicitly RGB
    description="Z=4, SPP=3, uint16, Contig, Tile=16x16",
)

# --- Test Case 3: Time (2) x Channel (3) Stack, SPP=1 ---
# Default IFD order: C fastest, then T (alphabetical: c, t)
write_tiff(
    filename=OUTPUT_DIR / "stack_t2_c3_spp1_float32.tif",
    base_shape=(BASE_HEIGHT, BASE_WIDTH),
    dtype=np.float32,
    stack_dims={"t": 2, "c": 3},
    description="T=2, C=3, SPP=1, float32, Contig, Tile=16x16. Default IFD order (C fastest)",
)

# --- Test Case 4: Time (2) x Channel (3) Stack, SPP=1, T fastest in file ---
# Specify IFD sequence order: ['c', 't'] means C varies slowest, T fastest
write_tiff(
    filename=OUTPUT_DIR / "stack_c3_t2_spp1_t_fastest.tif",
    base_shape=(BASE_HEIGHT, BASE_WIDTH),
    dtype=np.uint8,
    stack_dims={"c": 3, "t": 2},
    ifd_sequence_order=["c", "t"],  # C slowest, T fastest
    description="C=3, T=2, SPP=1, uint8, Contig, Tile=16x16. T fastest IFD order",
)

# --- Test Case 5: Stripped Z-Stack (3 planes), SPP=1 ---
write_tiff(
    filename=OUTPUT_DIR / "stack_z3_spp1_uint8_stripped.tif",
    base_shape=(BASE_HEIGHT, BASE_WIDTH),
    dtype=np.uint8,
    stack_dims={"z": 3},
    tile_shape=None,  # Stripped
    description="Z=3, SPP=1, uint8, Contig, Stripped",
)

# --- Test Case 6: Single IFD, but SPP=4 (RGBA example) ---
write_tiff(
    filename=OUTPUT_DIR / "single_spp4_rgba_uint8.tif",
    base_shape=(BASE_HEIGHT, BASE_WIDTH),
    dtype=np.uint8,
    stack_dims=None,  # Single IFD
    spp=4,
    planar_config_str="contig",
    photometric="rgb",  # Use 'rgb'
    extrasamples=(1,),  # Specify associated alpha
    description="Single IFD, SPP=4 (RGBA), uint8, Contig, Tile=16x16",
)

# --- Test Case 7: Z (2) x T (3) stack, SPP=1, Different Dtype ---
# IFD order Z, T (T fastest)
write_tiff(
    filename=OUTPUT_DIR / "stack_z2_t3_spp1_int16.tif",
    base_shape=(BASE_HEIGHT, BASE_WIDTH),
    dtype=np.int16,
    stack_dims={"z": 2, "t": 3},
    ifd_sequence_order=["z", "t"],  # T fastest
    description="Z=2, T=3, SPP=1, int16, Contig, Tile=16x16. T fastest IFD order",
)

# --- Test Case 8: single‑image, Zstd‑compressed ---
write_tiff(
    filename=OUTPUT_DIR / "single_zstd_uint8.tif",
    base_shape=(BASE_HEIGHT, BASE_WIDTH),
    dtype=np.uint8,
    stack_dims=None,
    compression="zstd",
    description="Single IFD, uint8, Zstd compression, Tile=16x16",
)

# --- Test Case 8: single‑image, zlib‑compressed ---
write_tiff(
    filename=OUTPUT_DIR / "single_zlib_uint8.tif",
    base_shape=(BASE_HEIGHT, BASE_WIDTH),
    dtype=np.uint8,
    stack_dims=None,
    compression="zlib",
    description="Single IFD, uint8, zlib compression, Tile=16x16",
)

logging.info(f"Finished generating TIFF files in {OUTPUT_DIR}")

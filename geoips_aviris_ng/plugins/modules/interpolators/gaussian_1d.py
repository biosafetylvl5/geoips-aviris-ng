import logging
from collections.abc import Callable
from functools import wraps

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# Module metadata
interface = "interpolators"
family = "1d"
name = "1d_gaussian"

# Configure logging
logger = logging.getLogger(__name__)

# Constants
SPATIAL_DIMS = frozenset(["x", "lon", "longitude"])
AVAILABLE_METHODS = frozenset(["gp", "gaussian_filter"])
MIN_POINTS_REQUIRED = 2
DENSE_GRID_SIZE = 1000
GAUSSIAN_SIGMA = 2.0


# Decorators for logging and validation
def log_function_call(func: Callable) -> Callable:
    """Decorator to log function calls."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"Calling {func_name}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Successfully completed {func_name}")
            return result
        except Exception:
            logger.exception(f"Error in {func_name}")
            raise

    return wrapper


def validate_method(method: str) -> None:
    """Validate interpolation method."""
    if method not in AVAILABLE_METHODS:
        raise ValueError(
            f"Unknown method: {method}. Available: {list(AVAILABLE_METHODS)}",
        )


def validate_1d_data(data_array: xr.DataArray) -> None:
    """Validate that data is 1D."""
    if len(data_array.dims) != 1:
        raise ValueError(
            f"Only 1D data supported. Got {len(data_array.dims)}D data with dims: {data_array.dims}",
        )


# Pure transformation functions
# @log_function_call
def dict_to_sorted_arrays(
    points_dict: dict[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert dictionary to sorted arrays."""
    if not points_dict:
        logger.warning("Empty points dictionary provided")
        return np.array([]), np.array([])

    x_vals = np.array(list(points_dict.keys()))
    y_vals = np.array(list(points_dict.values()))
    sort_idx = np.argsort(x_vals)

    logger.debug(f"Converted {len(points_dict)} points to sorted arrays")
    return x_vals[sort_idx], y_vals[sort_idx]


# @log_function_call
def arrays_to_dict(x_vals: np.ndarray, y_vals: np.ndarray) -> dict[float, float]:
    """Convert arrays back to dictionary."""
    result = {float(x): float(y) for x, y in zip(x_vals, y_vals, strict=False)}
    logger.debug(f"Converted arrays to dictionary with {len(result)} points")
    return result


def reshape_for_sklearn(arr: np.ndarray) -> np.ndarray:
    """Reshape array for sklearn compatibility."""
    return arr.reshape(-1, 1)


def clean_data(x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove NaN values from data."""
    valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
    x_clean = x_vals[valid_mask]
    y_clean = y_vals[valid_mask]

    removed_count = len(x_vals) - len(x_clean)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} NaN values from data")

    return x_clean, y_clean


def validate_sufficient_points(x_vals: np.ndarray, context: str = "") -> None:
    """Validate sufficient points for interpolation."""
    if len(x_vals) < MIN_POINTS_REQUIRED:
        msg = f"Insufficient points for interpolation: {len(x_vals)} < {MIN_POINTS_REQUIRED}"
        if context:
            msg += f" ({context})"
        logger.warning(msg)
        raise ValueError(msg)


# Interpolation strategies
# @log_function_call
def gp_interpolate(
    x_original: np.ndarray,
    y_original: np.ndarray,
    target_x: np.ndarray,
) -> np.ndarray:
    """Gaussian Process interpolation."""
    logger.info(
        f"GP interpolation: {len(x_original)} input points → {len(target_x)} target points",
    )

    kernel = ConstantKernel(1.0, (1e-9, 1e9)) * RBF(1.0, (1e-9, 1e5))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(reshape_for_sklearn(x_original), y_original)
    y_pred, _ = gp.predict(reshape_for_sklearn(target_x), return_std=True)

    logger.debug("GP interpolation completed successfully")
    return y_pred


# @log_function_call
def gaussian_filter_interpolate(
    x_original: np.ndarray,
    y_original: np.ndarray,
    target_x: np.ndarray,
) -> np.ndarray:
    """Gaussian filter interpolation."""
    logger.info(
        f"Gaussian filter interpolation: {len(x_original)} input points → {len(target_x)} target points",
    )

    # Create interpolation pipeline
    cubic_interp = interp1d(
        x_original,
        y_original,
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate",
    )

    # Dense grid for filtering
    x_range = (
        min(np.min(x_original), np.min(target_x)),
        max(np.max(x_original), np.max(target_x)),
    )
    x_dense = np.linspace(*x_range, DENSE_GRID_SIZE)

    # Apply transformations in sequence
    y_dense = cubic_interp(x_dense)
    y_smooth = gaussian_filter1d(y_dense, sigma=GAUSSIAN_SIGMA)

    # Final interpolation
    final_interp = interp1d(
        x_dense,
        y_smooth,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    result = final_interp(target_x)

    logger.debug("Gaussian filter interpolation completed successfully")
    return result


# Strategy mapping
INTERPOLATION_STRATEGIES: dict[str, Callable] = {
    "gp": gp_interpolate,
    "gaussian_filter": gaussian_filter_interpolate,
}


# @log_function_call
def gaussian_interpolate_points(
    points_dict: dict[float, float],
    target_x: list | np.ndarray,
    method: str = "gp",
) -> dict[float, float]:
    """
    Interpolate points using Gaussian-based methods.

    Parameters
    ----------
    points_dict : dict
        Dictionary where keys are x-coordinates and values are y-coordinates
    target_x : array-like
        New x-coordinates where you want to interpolate
    method : str
        'gp' for Gaussian Process, 'gaussian_filter' for Gaussian filtering

    Returns
    -------
    dict : Dictionary mapping target_x values to interpolated y values
    """
    validate_method(method)
    logger.info(f"Starting {method} interpolation with {len(points_dict)} input points")

    # Data preparation pipeline
    x_original, y_original = dict_to_sorted_arrays(points_dict)
    x_clean, y_clean = clean_data(x_original, y_original)
    validate_sufficient_points(x_clean, "after cleaning")

    target_x_array = np.array(target_x)
    logger.debug(f"Target points: {len(target_x_array)}")

    # Apply interpolation strategy
    interpolation_func = INTERPOLATION_STRATEGIES[method]
    y_pred = interpolation_func(x_clean, y_clean, target_x_array)

    # Convert back to dictionary
    return arrays_to_dict(target_x_array, y_pred)


def get_non_spatial_dims(data_array: xr.DataArray) -> list[str]:
    """Get non-spatial dimensions from data array."""
    return [dim for dim in data_array.dims if dim not in SPATIAL_DIMS]


# @log_function_call
def select_array_slice(data_array: xr.DataArray, array_num: int) -> xr.DataArray:
    """Select specific slice from data array based on array_num."""
    time_dims = get_non_spatial_dims(data_array)

    if time_dims and len(data_array[time_dims[0]]) > array_num:
        logger.debug(f"Selecting slice {array_num} from dimension {time_dims[0]}")
        return data_array.isel({time_dims[0]: array_num})
    else:
        logger.warning(f"array_num {array_num} out of range, using full array")
        return data_array


def get_target_coordinates(
    coord_dim: str,
    output_template: xr.Dataset,
    fallback_coords: np.ndarray,
) -> np.ndarray:
    """Get target coordinates from output template or fallback."""
    if coord_dim in output_template.coords:
        logger.debug(f"Using target coordinates from template: {coord_dim}")
        return output_template.coords[coord_dim].values
    else:
        logger.warning(
            f"Coordinate '{coord_dim}' not found in output template, using input coordinates",
        )
        return fallback_coords


# @log_function_call
def interpolate_1d_variable(
    input_var: xr.DataArray,
    output_template: xr.Dataset,
    method: str,
) -> np.ndarray:
    """Perform 1D interpolation on a variable."""
    validate_1d_data(input_var)
    validate_method(method)

    coord_dim = input_var.dims[0]
    logger.info(f"Interpolating variable along dimension: {coord_dim}")

    # Extract data
    input_x = input_var.coords[coord_dim].values
    input_y = input_var.values

    # Clean and validate data
    input_x_clean, input_y_clean = clean_data(input_x, input_y)

    if len(input_x_clean) < MIN_POINTS_REQUIRED:
        logger.warning("Not enough valid points for interpolation, returning NaN array")
        output_size = len(output_template.coords.get(coord_dim, input_x))
        return np.full(output_size, np.nan)

    # Get target coordinates
    target_x = get_target_coordinates(coord_dim, output_template, input_x)

    # Create points dictionary and interpolate
    points_dict = {
        float(x): float(y) for x, y in zip(input_x_clean, input_y_clean, strict=False)
    }
    result_dict = gaussian_interpolate_points(points_dict, target_x, method=method)

    # Convert back to array
    return np.array([result_dict[float(x)] for x in target_x])


# @log_function_call
def create_output_dataarray(
    data: np.ndarray,
    output_template: xr.Dataset,
    var_name: str,
    input_var: xr.DataArray,
) -> xr.DataArray:
    """Create output DataArray with proper coordinates and attributes."""
    coord_dim = input_var.dims[0]

    # Determine output coordinates - prioritize output_template coordinates
    if coord_dim in output_template.coords:
        output_coords = {coord_dim: output_template.coords[coord_dim]}
        logger.debug(f"Using output template coordinates for {coord_dim}")
    else:
        output_coords = {coord_dim: input_var.coords[coord_dim]}
        logger.debug(f"Using input coordinates for {coord_dim}")

    logger.debug(f"Creating output DataArray for {var_name} with dimension {coord_dim}")

    return xr.DataArray(
        data,
        dims=[coord_dim],
        coords=output_coords,
        attrs=input_var.attrs.copy(),
        name=var_name,
    )


def process_variable(
    var_name: str,
    input_xarray: xr.Dataset,
    output_template: xr.Dataset,
    array_num: int | None,
    method: str,
) -> xr.DataArray | None:
    """Process a single variable through the interpolation pipeline."""
    if var_name not in input_xarray.data_vars:
        logger.warning(f"Variable '{var_name}' not found in input dataset")
        return None

    logger.info(f"Processing variable: {var_name}")

    # Get and potentially slice input variable
    input_var = input_xarray[var_name]
    if array_num is not None:
        input_var = select_array_slice(input_var, array_num)

    # Perform interpolation
    interpolated_data = interpolate_1d_variable(input_var, output_template, method)

    # Create output DataArray
    return create_output_dataarray(
        interpolated_data,
        output_template,
        var_name,
        input_var,
    )


def call(
    area_def,
    input_xarray: xr.Dataset,
    output_xarray: xr.Dataset,
    varlist: list[str],
    array_num: int | None = None,
    method: str = "gp",
) -> xr.Dataset:
    """
    Call function to perform 1D Gaussian interpolation on xarray datasets.

    Parameters
    ----------
    area_def : object
        Area definition object (not used in 1D interpolation)
    input_xarray : xr.Dataset
        Input dataset containing source data
    output_xarray : xr.Dataset
        Output dataset template with target coordinates
    varlist : List[str]
        List of variable names to interpolate
    array_num : Optional[int]
        Specific array/time index to process (if None, process all)
    method : Optional[str]
        Interpolation method ('gp' or 'gaussian_filter')

    Returns
    -------
    xr.Dataset : Interpolated output dataset
    """
    validate_method(method)
    logger.info(f"Starting 1D Gaussian interpolation with method: {method}")
    logger.info(f"Processing {len(varlist)} variables: {varlist}")

    # Create result dataset
    result_dataset = output_xarray.copy(deep=True)

    # Process each variable
    processed_count = 0
    for var_name in varlist:
        result_var = process_variable(
            var_name,
            input_xarray,
            output_xarray,
            array_num,
            method,
        )
        if result_var is not None:
            result_dataset[var_name] = result_var
            processed_count += 1

    logger.info(f"Successfully processed {processed_count}/{len(varlist)} variables")
    return result_dataset


# Example usage:
if __name__ == "__main__":
    # Configure logging for examples
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    # Example data: dictionary of points a -> b
    points = {0.0: 1.0, 1.0: 2.5, 2.0: 3.2, 3.0: 2.8, 4.0: 4.1, 5.0: 5.0}

    # Target x-coordinates
    target_coordinates = [0.5, 1.5, 2.5, 3.5, 4.5]

    print("Original points (a -> b):")
    for a, b in points.items():
        print(f"  {a} -> {b}")

    # Interpolate with Gaussian Process
    gp_results = gaussian_interpolate_points(points, target_coordinates, method="gp")
    print("\nGaussian Process interpolated points (c -> d):")
    for c, d in gp_results.items():
        print(f"  {c} -> {d:.3f}")

    # Interpolate with Gaussian Filter
    filter_results = gaussian_interpolate_points(
        points,
        target_coordinates,
        method="gaussian_filter",
    )
    print("\nGaussian Filter interpolated points (c -> d):")
    for c, d in filter_results.items():
        print(f"  {c} -> {d:.3f}")

    # Example with xarray
    print("\n=== XArray Example ===")

    # Create sample 1D input dataset
    input_data = xr.Dataset(
        {
            "temperature": (["x"], np.random.rand(10) * 30),
            "pressure": (["x"], np.random.rand(10) * 1000 + 900),
        },
        coords={"x": np.linspace(0, 100, 10)},
    )

    # Create output template with different resolution
    output_template = xr.Dataset(
        {
            "temperature": (["x"], np.full(15, np.nan)),
            "pressure": (["x"], np.full(15, np.nan)),
        },
        coords={
            "x": np.linspace(0, 100, 15),  # Higher resolution
        },
    )

    # Call the GeoIPS entrypoint
    result = call(
        area_def=None,
        input_xarray=input_data,
        output_xarray=output_template,
        varlist=["temperature", "pressure"],
        array_num=None,
        method="gp",
    )

    print("1D Interpolation completed successfully!")
    print(f"Input shape: {input_data.dims}")
    print(f"Output shape: {result.dims}")
    print(f"Variables: {list(result.data_vars.keys())}")

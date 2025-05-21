"""
AVIRIS-NG Hyperspectral Imagery Reader for GeoIPS.

This module provides functionality to read and process data from the Airborne Visible 
InfraRed Imaging Spectrometer-Next Generation (AVIRIS-NG) instrument. AVIRIS-NG is an 
airborne hyperspectral sensor operated by NASA Jet Propulsion Laboratory that collects 
data across 224-480 spectral bands ranging from 380 nm to 2500 nm.

The reader supports:
    - Level 1 radiance data (units: W/m^2/sr/nm)
    - Level 2 reflectance data (unitless)
    - Level 2 corrected reflectance data
    - Level 2 water vapor products (units: cm)

AVIRIS-NG data is provided in ENVI format (.img files with accompanying .hdr files).
This reader uses GDAL to access the data and converts it to xarray Datasets for 
further processing in the GeoIPS framework.

Architecture Overview:
---------------------

1. Entry Points:
   - call(): Main entry point that interfaces with GeoIPS
   - _call_single_time(): Processes a single time period

2. Core Reading Functions:
   - read_aviris_file(): Reads a single AVIRIS-NG file and converts to xarray
   - get_metadata(): Extracts metadata from the file
   - read_band_data(): Reads individual spectral bands

3. Helper Functions:
   - get_datetime_from_filename(): Parses datetime from filenames
   - determine_data_type(): Identifies data level and type from filenames
   - get_band_info(): Extracts band wavelength information
   - determine_bands_to_read(): Maps requested wavelengths to band indices
   - create_variable_name(): Generates standardized variable names
   - set_metadata(): Populates metadata in the xarray Dataset

Data Flow:
---------
1. The GeoIPS framework calls the `call()` function with a list of filenames.
2. The call() function delegates to _call_single_time() for each time period.
3. For each file:
   a. The file is opened using GDAL
   b. Metadata is extracted and parsed
   c. If metadata_only=True, only metadata is returned
   d. Otherwise, requested bands are identified and read
   e. Each band is converted to an xarray DataArray with appropriate attributes
   f. All bands are combined into a single xarray Dataset
4. Results are returned as a dictionary of xarray Datasets keyed by product type

Band Processing:
--------------
The reader handles the spectral AVIRIS-NG data by:
1. Extracting wavelength information for each band from metadata
2. Classifying bands into regions of the electromagnetic spectrum
3. Creating standardized variable names based on wavelength and region
4. Allowing users to request specific wavelengths, finding the closest matches


Example:
-------

>>> from geoips.plugins.modules.readers.aviris_ng import call
>>> 
>>> # Read all bands
>>> data = call(['path/to/aviris_ng_file.img'])
>>> 
>>> # Read only specific wavelengths (in nm)
>>> data = call(['path/to/aviris_ng_file.img'], chans=[450, 550, 650])
>>> 
>>> # Read only metadata
>>> metadata = call(['path/to/aviris_ng_file.img'], metadata_only=True)
>>> 
>>> # Force interpretation as a specific data type
>>> data = call(['path/to/aviris_ng_file.img'], force_type='rfl')

Notes:
    This reader is designed for GeoIPS and follows its plugin architecture.
    It returns data in the standard GeoIPS format (dictionary of xarray Datasets).
"""

import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from geoips.interfaces import readers
from osgeo import gdal

# Define required plugin attributes
interface = "readers"
family = "standard"
name = "aviris_ng"

LOG = logging.getLogger(__name__)

# Band descriptions
band_dictionary = {
    "visible-violet": {"lower": 375, "upper": 450, "color": "violet"},
    "visible-blue": {"lower": 450, "upper": 485, "color": "blue"},
    "visible-cyan": {"lower": 485, "upper": 500, "color": "cyan"},
    "visible-green": {"lower": 500, "upper": 565, "color": "green"},
    "visible-yellow": {"lower": 565, "upper": 590, "color": "yellow"},
    "visible-orange": {"lower": 590, "upper": 625, "color": "orange"},
    "visible-red": {"lower": 625, "upper": 740, "color": "red"},
    "near-infrared": {"lower": 740, "upper": 1100, "color": "gray"},
    "shortwave-infrared": {"lower": 1100, "upper": 2500, "color": "white"},
}


def get_datetime_from_filename(fname):
    """
    Extract datetime from AVIRIS-NG filename.

    AVIRIS-NG filenames follow the pattern: angYYYYMMDDtHHNNSS

    Parameters
    ----------
    fname : str
        Filename to parse

    Returns
    -------
    datetime
        Extracted datetime object
        
    Raises
    ------
    ValueError
        If datetime cannot be extracted from the filename
    """
    # Extract the basename without path and extension
    basename = Path(fname).name

    # Extract date pattern using regex
    match = re.search(r"ang(\d{8})t(\d{6})", basename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)

        # Parse into datetime
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])

        return datetime(year, month, day, hour, minute, second)
    else:
        error_msg = f"Could not extract datetime from filename: {basename}"
        LOG.critical(error_msg)
        raise ValueError(error_msg)


def determine_data_type(fname):
    """
    Determine the type of AVIRIS-NG data based on the filename.

    Parameters
    ----------
    fname : str
        Path to the AVIRIS-NG data file

    Returns
    -------
    tuple
        (data_level, data_type, product_version)
        data_level: 'l1' or 'l2'
        data_type: 'rad' (radiance), 'rfl' (reflectance),
                   'corr' (corrected reflectance), or 'h2o' (water)
        product_version: version string (e.g., 'v2p5')
    """
    basename = Path(fname).name

    # Determine data level and type
    if "rfl" in basename:
        data_level = "l2"
        if "_corr_" in basename:
            data_type = "corr"
        elif "_h2o_" in basename:
            data_type = "h2o"
        else:
            data_type = "rfl"
    else:
        data_level = "l1"
        data_type = "rad"

    # Extract product version if available
    version_match = re.search(r"_v(\d+p\d+)", basename)
    if version_match:
        product_version = version_match.group(1)
    else:
        product_version = "unknown"
        LOG.warning(
            "Unable to extract product version from filename. "
            + "Using 'unknown' as default.",
        )

    return data_level, data_type, product_version


def get_band_info(img):
    """
    Get band information from AVIRIS-NG image.

    Parameters
    ----------
    img : gdal.Dataset
        GDAL dataset for the AVIRIS-NG image

    Returns
    -------
    pandas.DataFrame
        DataFrame with band information
    """

    # Function to classify bands
    def classifier(band):
        for region, limits in band_dictionary.items():
            if limits["lower"] <= band <= limits["upper"]:
                return region
        return None

    # Get band metadata
    metadata = img.GetMetadata()

    # Lists of band numbers, band centers, and em classes
    band_numbers = []
    band_centers = []

    try:
        # Process metadata to extract band information
        for key, value in metadata.items():
            if key != "wavelength_units" and "_" in key:
                try:
                    band_num = int(key.split("_")[1])
                    band_numbers.append(band_num)

                    # Handle different metadata formats
                    if " " in value:
                        wavelength = float(value.split(" ")[0])
                    else:
                        wavelength = float(value)

                    band_centers.append(wavelength)
                except (ValueError, IndexError):
                    LOG.warning(f"Could not parse band metadata: {key}={value}")
    except Exception:
        LOG.critical("Error processing band metadata.")
        raise

    em_regions = [classifier(b) for b in band_centers]

    # Create DataFrame describing bands
    bands = pd.DataFrame(
        {
            "Band number": band_numbers,
            "Band center (nm)": band_centers,
            "EM region": em_regions,
        },
        index=band_numbers,
    ).sort_index()

    return bands


def create_variable_name(wavelength, data_type):
    """
    Create standardized variable names for spectral bands.

    Parameters
    ----------
    wavelength : float
        Wavelength in nanometers
    data_type : str
        Type of data ('rad', 'rfl', 'corr', 'h2o')

    Returns
    -------
    str
        Variable name for the dataset
    """
    # Special handling for water vapor data
    h2o_band_names = ["column_water_vapor", "liquid_h2o_absorption", "ice_absorption"]

    if data_type == "h2o":
        if wavelength in [1, 2, 3]:
            return h2o_band_names[wavelength - 1]
        else:
            return f"h2o_band_{wavelength:.1f}"

    # For spectral data, use wavelength-based naming
    region = "other"

    # Find the band region using the dictionary
    for band, info in band_dictionary.items():
        if info["lower"] <= wavelength <= info["upper"]:
            # Extract just the main color name for the variable
            if band.startswith("visible-"):
                region = info["color"]
            elif band == "near-infrared":
                region = "nir"
            elif band == "shortwave-infrared":
                region = "swir"
            break

    # Create variable name with region, wavelength, and data type
    return f"{region}_{int(wavelength)}nm_{data_type}"


def get_metadata(img, fname, force_type=None):
    """
    Extract metadata from an AVIRIS image.

    Parameters
    ----------
    img : gdal.Dataset
        GDAL dataset object for the AVIRIS image
    fname : str
        Path to the AVIRIS-NG data file
    force_type : str, optional
        Force data type interpretation ('corr', 'h2o', 'rad', or 'rfl')

    Returns
    -------
    dict
        Dictionary containing all metadata
    """
    # Validate force_type if provided
    if force_type not in ["corr", "h2o", "rad", "rfl", None]:
        raise ValueError(
            f"Invalid force_type: {force_type}. "
            + "Must be one of: 'corr', 'h2o', 'rad', 'rfl'."
            + "Pass 'None' or do not pass the argument to auto-detect type.",
        )

    # Get basic metadata
    nbands = img.RasterCount
    nrows = img.RasterYSize
    ncols = img.RasterXSize

    # Get geotransform information
    xmin, xres, xrot, ymax, yrot, yres = img.GetGeoTransform()

    # Generate coordinate arrays
    x_coords = np.array([xmin + i * xres for i in range(ncols)])
    y_coords = np.array([ymax + i * yres for i in range(nrows)])

    # Extract datetime from filename
    file_datetime = get_datetime_from_filename(fname)

    data_level, data_type, product_version = determine_data_type(fname)

    # Override data type if forced
    if force_type is not None:
        data_type = force_type
        LOG.info(f"Forcing data type to: {force_type}")

    # Get projection information
    projection = img.GetProjection() if img.GetProjection() else None

    # Get band information
    bands_info = get_band_info(img)

    with open(fname) as f:
        data_ignore_value = False
        for ln in f.readlines():
            if " = " in ln and "data ignore value" in ln:
                data_ignore_value = int(ln.split(" = ")[1])
                break
        if not data_ignore_value:
            LOG.critical("Could not identify data ignore value.")
            raise ValueError("Could not identify data ignore value in file.")

    # Return all metadata as a dictionary
    return {
        "nbands": nbands,
        "nrows": nrows,
        "ncols": ncols,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "file_datetime": file_datetime,
        "data_level": data_level,
        "data_type": data_type,
        "product_version": product_version,
        "projection": projection,
        "bands_info": bands_info,
        "data_ignore_value": data_ignore_value,
    }


def set_metadata(dataset, metadata, fname):
    """
    Set metadata attributes in the xarray Dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to update with metadata
    metadata : dict
        Dictionary containing metadata from get_metadata
    fname : str
        Path to the AVIRIS-NG data file

    Returns
    -------
    xarray.Dataset
        Updated dataset with metadata
    """
    # Create a new dataset to avoid modifying the input
    new_dataset = dataset.copy()

    # Set required attributes
    new_dataset.attrs["source_name"] = "aviris_ng"
    new_dataset.attrs["platform_name"] = "aircraft"
    new_dataset.attrs["data_provider"] = "nasa_jpl"
    new_dataset.attrs["start_datetime"] = metadata["file_datetime"]
    new_dataset.attrs["end_datetime"] = metadata[
        "file_datetime"
    ]  # Not sure how to get this
    new_dataset.attrs["interpolation_radius_of_influence"] = (
        20  # 20m (a guess for 5m resolution data)
    )

    # Add data-specific attributes
    new_dataset.attrs["data_level"] = metadata["data_level"]
    new_dataset.attrs["data_type"] = metadata["data_type"]
    new_dataset.attrs["product_version"] = metadata["product_version"]
    new_dataset.attrs["data_ignore_value"] = metadata["data_ignore_value"]

    # Optional attributes
    new_dataset.attrs["source_file_names"] = [Path(fname).name]
    new_dataset.attrs["sample_distance_km"] = 0.005  # 5m resolution in km

    # Add projection information if available
    if metadata["projection"]:
        new_dataset.attrs["projection"] = metadata["projection"]

    # Add coordinate information
    new_dataset.coords["y"] = metadata["y_coords"]
    new_dataset.coords["x"] = metadata["x_coords"]

    # Create 2D arrays for latitude and longitude
    lons, lats = np.meshgrid(metadata["x_coords"], metadata["y_coords"])

    # Add required lat/lon variables
    new_dataset["latitude"] = (("y", "x"), lats)
    new_dataset["longitude"] = (("y", "x"), lons)

    # Add time dimension
    time_var = np.full(
        (metadata["nrows"], metadata["ncols"]),
        metadata["file_datetime"],
    )

    new_dataset["time"] = (("y", "x"), time_var)

    return new_dataset


def read_band_data(img, band_idx, bands_info, data_type, data_ignore_value):
    """
    Read a single band from the AVIRIS image.

    Parameters
    ----------
    img : gdal.Dataset
        GDAL dataset object for the AVIRIS image
    band_idx : int
        Band index to read
    bands_info : pandas.DataFrame
        DataFrame containing band information
    data_type : str
        Data type of the image ('rad', 'rfl', 'corr', 'h2o')
    data_ignore_value : int
        Value to be treated as missing/invalid data

    Returns
    -------
    tuple
        (variable_name, band_data, wavelength, attributes)
    """
    # Read band data
    band_data = img.GetRasterBand(band_idx).ReadAsArray()

    # Get wavelength for this band
    if band_idx in bands_info.index:
        wavelength = bands_info.loc[band_idx, "Band center (nm)"]
    else:
        # Use band index as wavelength if not found in metadata
        wavelength = float(band_idx)
        LOG.warning(f"Band {band_idx} not found in metadata, using index as wavelength. Data may be incorrect.")

    # Handle fill values
    band_data = np.where(band_data == data_ignore_value, np.nan, band_data)

    # Create variable name
    var_name = create_variable_name(wavelength, data_type)

    # Create attributes dictionary
    attributes = {"wavelength": wavelength, "band_number": band_idx}

    # Set appropriate units based on data type
    if data_type == "rad":
        attributes["units"] = "W/m^2/sr/nm"
    elif data_type in ["rfl", "corr"]:
        attributes["units"] = "reflectance"
    elif data_type == "h2o":
        attributes["units"] = "cm"  # Column water vapor or Absorption path

    return var_name, band_data, wavelength, attributes


def determine_bands_to_read(chans, bands_info, nbands):
    """
    Determine which bands to read based on user input.

    Parameters
    ----------
    chans : list or None
        List of specific wavelengths (in nm) to read
    bands_info : pandas.DataFrame
        DataFrame containing band information
    nbands : int
        Total number of bands in the image
    Returns
    -------
    list
        List of band indices to read
    """
    if not chans:
        # Read all bands by default
        return list(range(1, nbands + 1))

    # Find the band numbers closest to the requested wavelengths
    band_indices = []
    for chan in chans:
        try:
            # Convert to float to ensure we're treating it as a wavelength
            chan_float = float(chan)
            closest_band = bands_info.iloc[
                (bands_info["Band center (nm)"] - chan_float).abs().argsort()[0]
            ]
            band_indices.append(int(closest_band["Band number"]))
            LOG.info(f"Using band {int(closest_band['Band number'])} for requested wavelength {chan_float} nm")
        except (ValueError, TypeError):
            LOG.warning(f"Could not interpret channel as wavelength: {chan}")

    return band_indices


def read_aviris_file(fname, chans=None, metadata_only=False, force_type=None):
    """
    Read AVIRIS-NG data from a single file.

    Parameters
    ----------
    fname : str
        Path to the AVIRIS-NG data file (ENVI format)
    chans : list, optional
        List of specific wavelengths (in nm) to read
    metadata_only : bool, optional
        If True, only read metadata without loading full dataset
    force_type : str, optional
        Force data type interpretation ('corr', 'h2o', 'rad', or 'rfl')

    Returns
    -------
    xarray.Dataset
        Dataset containing the AVIRIS-NG data

    Raises
    ------
    ValueError
        If the file cannot be opened or processed
    """
    # Create empty dataset
    dataset = xr.Dataset()

    # Open the ENVI file with GDAL
    LOG.info(f"Opening AVIRIS-NG file: {fname}")
    img = gdal.Open(fname)

    if img is None:
        LOG.critical(f"Can not open file: {fname}")
        raise ValueError(f"GDAL could not open file: {fname}")

    # Get metadata
    metadata = get_metadata(img, fname, force_type)

    # Set metadata in the dataset
    dataset = set_metadata(dataset, metadata, fname)

    if metadata_only:
        LOG.debug("metadata_only requested, returning without reading data")
        return dataset

    # Determine which bands to read
    band_indices = determine_bands_to_read(
        chans,
        metadata["bands_info"],
        metadata["nbands"],
    )

    # Read selected bands
    for band_idx in band_indices:
        try:
            var_name, band_data, wavelength, attributes = read_band_data(
                img,
                band_idx,
                metadata["bands_info"],
                metadata["data_type"],
                metadata["data_ignore_value"],
            )

            # Add band to dataset
            dataset[var_name] = (("y", "x"), band_data)

            # Add attributes to the variable
            for key, value in attributes.items():
                dataset[var_name].attrs[key] = value

        except Exception:
            LOG.exception(f"Error reading band {band_idx}")
            raise

    return dataset


def _call_single_time(  # noqa: PLR0913
    fnames,
    metadata_only=False,
    chans=None,
    area_def=None,
    self_register=False,
    force_type=None,
):
    """
    Process a single file or group of files for one time period.

    Parameters
    ----------
    fnames : list
        List of strings, full paths to files
    metadata_only : bool, default=False
        Return before reading data if True
    chans : list, default=None
        List of desired wavelengths (in nm)
    area_def : pyresample.AreaDefinition, default=None
        Specify region to read
    self_register : bool, default=False
        Register all data to a specified dataset
    force_type : str, default=None
        Force data type interpretation ('corr', 'h2o', 'rad', or 'rfl')

    Returns
    -------
    dict
        Dictionary of xarray.Dataset objects
    """
    if self_register:
        LOG.warning(
            f"self_register was passed with non-default value '{self_register}'. "
            "However self_register is not implemented for this reader."
        )

    if area_def:
        LOG.warning(
            "area_def was provided but is not implemented for this reader. "
            "Please manually sector if you want this functionality."
        )

    if not fnames:
        LOG.error("No filenames provided")
        return {}

    # Initialize result dictionary
    result = {}

    # Process each file in the list
    for fname in fnames:
        try:
            dataset = read_aviris_file(
                fname,
                chans=chans,
                metadata_only=metadata_only,
                force_type=force_type,
            )

            # Determine appropriate key for the dataset based on data type
            data_level = dataset.attrs.get("data_level", "unknown")
            data_type = dataset.attrs.get("data_type", "unknown")

            if data_level == "l1":
                key = "AVIRIS-NG-L1-RADIANCE"
            elif data_level == "l2":
                if data_type == "corr":
                    key = "AVIRIS-NG-L2-REFLECTANCE"
                elif data_type == "h2o":
                    key = "AVIRIS-NG-L2-WATER"
                else:
                    key = "AVIRIS-NG-L2"
            else:
                key = "AVIRIS-NG"

            # Add dataset to result dictionary
            result[key] = dataset

            # Add metadata if not already present
            if "METADATA" not in result:
                result["METADATA"] = dataset[[]]

        except Exception:
            LOG.exception(f"Error processing file {fname}")
            raise

    return result


def call(  # noqa: PLR0913
    fnames,
    metadata_only=False,
    chans=None,
    area_def=None,
    self_register=False,
    force_type=None,
):
    """
    Read AVIRIS-NG data from one or more files.

    Parameters
    ----------
    fnames : list
        List of strings, full paths to files
    metadata_only : bool, default=False
        Return before reading data if True
    chans : list, default=None
        List of desired wavelengths (in nm)
    area_def : pyresample.AreaDefinition, default=None
        Specify region to read
    self_register : bool, default=False
        Register all data to a specified dataset
    force_type : str, default=None
        Force data type interpretation ('corr', 'h2o', 'rad', or 'rfl')

    Returns
    -------
    dict
        Dictionary of xarray.Dataset objects
    """
    return readers.read_data_to_xarray_dict(
        fnames,
        _call_single_time,
        metadata_only,
        chans,
        area_def,
        self_register,
        force_type=force_type,
    )

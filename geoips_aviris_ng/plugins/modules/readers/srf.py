"""
GeoIPS reader for Spectral Response Function (SRF) files.

This module provides a GeoIPS-compliant reader for Spectral Response Function (SRF)
files containing instrument channel spectral characteristics. SRF files define how
satellite instruments respond to electromagnetic radiation across different wavelengths.

File Format
-----------
SRF files are ASCII text files with two columns:
    - Column 1: Wavenumber values (cm^-1)
    - Column 2: Spectral response values (dimensionless, 0.0-1.0)

Expected filename format: INSTRUMENT_MODEL_DATE_CHANNEL.SRF
    - INSTRUMENT: Satellite instrument name (e.g., ABI, MODIS)
    - MODEL: Instrument model/version (e.g., PFM, FM1)
    - DATE: Date in format DDMmmYYYY (e.g., 10Mar2016)
    - CHANNEL: Channel identifier (e.g., CH01, CH02)

Data Processing
---------------
The reader performs several transformations to make spectral data compatible
with GeoIPS requirements:

1. **Unit Conversion**: Converts wavenumbers (cm^-1) to wavelengths (micrometers)
   using the relationship: wavelength = 10000 / wavenumber.

2. **Metadata Extraction**: Parses filename to extract instrument, model, date,
   and channel information for proper dataset attribution.

3. **CF Compliance**: Adds Climate and Forecast (CF) convention attributes to
   all variables for standardized metadata.

Output Structure
----------------
Returns xarray Dataset with:

**Data Variables**:
    - wavenumber: Original wavenumber values (cm^-1)
    - wavelength: Converted wavelength values (micrometers)
    - response: Spectral response function values (dimensionless)

**Attributes**:
    - Standard GeoIPS attributes (source_name, platform_name, etc.)
    - Instrument-specific metadata (instrument, channel, model)
    - CF-compliant variable attributes

Limitations
-----------
- Each SRF file may only contain data for a single instrument channel
- Does not include time/latitude/longitude coordinates (not applicable to spectral data)

Notes
-----
- Spatial subsetting (area_def) is not applicable to spectral data
- Channel selection (chans) parameter is ignored for single-channel files
- Self-registration is not meaningful for spectral data
- SRF files for many platforms can be found at https://cimss.ssec.wisc.edu/
- GOES Specific files can be found at https://cimss.ssec.wisc.edu/goes/calibration/
"""

import logging
import numpy as np
import xarray as xr
from datetime import datetime
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from geoips.interfaces import readers

# Define required plugin attributes
interface = "readers"
family = "standard"
name = "srf"

LOG = logging.getLogger(__name__)

# Constants for unit conversion
SPEED_OF_LIGHT = 2.99792458e10  # cm/s
WAVENUMBER_TO_WAVELENGTH_FACTOR = 1e4  # Convert cm^-1 to micrometers




def parse_srf_filename(filename: Union[str, os.PathLike]) -> Dict[str, Union[str, datetime]]:
    """Parse SRF filename to extract instrument metadata.

    Extracts instrument, model, date, and channel from standard SRF filename format.
    Expected format: INSTRUMENT_MODEL_DATE_CHANNEL.SRF

    Parameters
    ----------
    filename : str or PathLike
        SRF filename or path.

    Returns
    -------
    dict
        Metadata with keys: 'instrument', 'model', 'date', 'channel', 'platform_name'.

    Examples
    --------
    >>> metadata = parse_srf_filename("ABI_PFM_10Mar2016_CH01.SRF")
    >>> metadata['instrument']
    'ABI'
    >>> metadata['channel']
    'CH01'
    """
    LOG.debug(f"Parsing filename: {filename}")
    basename = Path(filename).stem

    # Pattern matches: INSTRUMENT_MODEL_DATE_CHANNEL
    pattern = r'([A-Za-z]+)_([A-Za-z0-9]+)_(\d{1,2}[A-Za-z]{3}\d{4})_([A-Za-z0-9]+)'
    match = re.match(pattern, basename)

    if match:
        instrument, model, date_str, channel = match.groups()
        LOG.debug(f"Parsed components: {instrument}, {model}, {date_str}, {channel}")

        # Parse date string (e.g., "10Mar2016")
        try:
            date_obj = datetime.strptime(date_str, "%d%b%Y")
            LOG.debug(f"Parsed date: {date_obj}")
        except ValueError:
            LOG.warning(f"Failed to parse date '{date_str}', using default")
            date_obj = datetime(999, 9, 9)

        return {
            'instrument': instrument,
            'model': model,
            'date': date_obj,
            'channel': channel,
            'platform_name': f"{instrument}_{model}"
        }
    else:
        LOG.warning(f"Filename '{basename}' does not match expected pattern, using defaults")
        return {
            'instrument': 'unknown',
            'model': 'unknown',
            'date': datetime(999, 9, 9),
            'channel': 'unknown',
            'platform_name': 'unknown'
        }


def convert_wavenumber_to_wavelength_in_um(wavenumbers: np.ndarray) -> np.ndarray:
    """Convert wavenumbers to wavelengths.

    Converts wavenumber values in cm^-1 to wavelength values in micrometers
    using the relationship: wavelength = 10000 / wavenumber.

    Parameters
    ----------
    wavenumbers : array_like
        Wavenumber values in cm^-1.

    Returns
    -------
    numpy.ndarray
        Wavelength values in micrometers.

    Notes
    -----
    Zero or negative wavenumbers will produce infinite or negative wavelengths.
    """
    LOG.debug(f"Converting {len(wavenumbers)} wavenumber values to wavelengths")
    wavenumbers = np.asarray(wavenumbers)

    # Check for problematic values
    if np.any(wavenumbers <= 0):
        LOG.warning("Found zero or negative wavenumbers, results may contain inf/nan")

    with np.errstate(divide='ignore', invalid='ignore'):
        wavelengths = WAVENUMBER_TO_WAVELENGTH_FACTOR / wavenumbers

    LOG.debug(f"Wavelength range: {np.min(wavelengths):.2f} - " +
              f"{np.max(wavelengths):.2f} μm")
    return wavelengths


def read_srf_data(filepath: Union[str, os.PathLike]) -> Tuple[np.ndarray, np.ndarray]:
    """Read numeric data from SRF file.

    Reads two-column ASCII data containing wavenumber and response values.

    Parameters
    ----------
    filepath : str or PathLike
        Path to SRF file.

    Returns
    -------
    wavenumbers : numpy.ndarray
        Wavenumber values from first column (cm^-1).
    responses : numpy.ndarray
        Response values from second column (dimensionless).

    Raises
    ------
    ValueError
        If file does not contain exactly two columns.
    IOError
        If file cannot be read.
    """
    LOG.info(f"Reading SRF data from: {filepath}")

    try:
        data = np.loadtxt(filepath, dtype=float)
        LOG.debug(f"Loaded data shape: {data.shape}")

        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f"Expected 2-column data, got shape {data.shape}")

        wavenumbers = data[:, 0]
        responses = data[:, 1]

        LOG.debug(f"Read {len(wavenumbers)} spectral data points")
        LOG.debug(f"Wavenumber range: {wavenumbers.min():.1f} - " +
                  f"{wavenumbers.max():.1f} cm^-1")
        LOG.debug(f"Response range: {responses.min():.3f} - {responses.max():.3f}")

        return wavenumbers, responses

    except Exception as e:
        LOG.error(f"Failed to read SRF data from {filepath}: {e}")
        raise e


def read_srf_file(fname: Union[str, os.PathLike],
                  metadata_only: bool = False) -> xr.Dataset:
    """Read spectral response function data from file.

    Reads SRF file and creates xarray Dataset with required GeoIPS attributes
    and spectral data variables.

    Parameters
    ----------
    fname : str or PathLike
        Path to SRF file.
    metadata_only : bool, default False
        If True, return only metadata without data arrays.

    Returns
    -------
    xarray.Dataset
        Dataset containing SRF data with required GeoIPS attributes.
    """
    LOG.info(f"Reading SRF file: {os.path.basename(fname)}")

    # Parse filename for metadata
    metadata = parse_srf_filename(fname)
    LOG.debug(f"Extracted metadata: {metadata}")

    # Create dataset with required GeoIPS attributes
    dataset = xr.Dataset()

    # Required attributes
    dataset.attrs["source_name"] = "srf"
    dataset.attrs["platform_name"] = metadata['platform_name']
    dataset.attrs["data_provider"] = "srf_file"
    dataset.attrs["start_datetime"] = metadata['date']
    dataset.attrs["end_datetime"] = metadata['date']
    dataset.attrs["interpolation_radius_of_influence"] = None

    # Optional attributes
    dataset.attrs["source_file_names"] = [os.path.basename(fname)]
    dataset.attrs["registered_dataset"] = False

    # SRF-specific attributes
    dataset.attrs["instrument"] = metadata['instrument']
    dataset.attrs["channel"] = metadata['channel']
    dataset.attrs["model"] = metadata['model']

    LOG.debug("Set dataset attributes")

    if metadata_only:
        LOG.info("Metadata-only requested, returning without data")
        return dataset

    # Read spectral data
    wavenumbers, responses = read_srf_data(fname)
    wavelengths = convert_wavenumber_to_wavelength_in_um(wavenumbers)

    # Add spectral data as 1D variables with spectral_point dimension
    dataset["wavenumber"] = (["spectral_point"], wavenumbers)
    dataset["wavelength"] = (["spectral_point"], wavelengths)
    dataset["response"] = (["spectral_point"], responses)

    # Set CF-compliant variable attributes
    dataset["wavenumber"].attrs = {
        "standard_name": "wavenumber",
        "long_name": "wavenumber",
        "units": "cm-1"
    }

    dataset["wavelength"].attrs = {
        "standard_name": "wavelength",
        "long_name": "wavelength",
        "units": "micrometers"
    }

    dataset["response"].attrs = {
        "standard_name": "spectral_response_function",
        "long_name": f"spectral response function for {metadata['channel']}",
        "units": "1",
        "valid_range": [0.0, 1.0]
    }

    LOG.info(f"Successfully created dataset with {len(wavenumbers)} spectral points")
    return dataset


def _call_single_time(fnames: List[Union[str, os.PathLike]],
                     metadata_only: bool = False,
                     chans: Optional[List[str]] = None,
                     area_def: Optional[object] = None,
                     self_register: bool = False) -> Dict[str, xr.Dataset]:
    """Process single SRF file.

    Reads one SRF file and returns dataset dictionary for GeoIPS processing.

    Parameters
    ----------
    fnames : list of str or PathLike
        List containing single SRF file path.
    metadata_only : bool, default False
        Return only metadata without data arrays.
    chans : list of str, optional
        Channel list (not applicable for SRF files).
    area_def : object, optional
        Area definition (not applicable for SRF files).
    self_register : bool, default False
        Self-registration flag (not applicable for SRF files).

    Returns
    -------
    dict
        Dictionary with dataset key and metadata dataset.

    Raises
    ------
    ValueError
        If more than one file provided.
    """
    if len(fnames) != 1:
        raise ValueError(f"SRF reader expects exactly one file, got {len(fnames)}")

    fname = fnames[0]
    LOG.info(f"Reading SRF file: {os.path.basename(fname)}")

    # Read the SRF data
    dataset = read_srf_file(fname, metadata_only=metadata_only)

    # Create dataset key using channel information
    channel = dataset.attrs.get('channel', 'unknown')
    dataset_key = f"SRF_{channel}"

    LOG.debug(f"Created dataset with key: {dataset_key}")
    return {dataset_key: dataset, "METADATA": dataset[[]]}


def call(fnames: List[Union[str, os.PathLike]],
         metadata_only: bool = False,
         chans: Optional[List[str]] = None,
         area_def: Optional[object] = None,
         self_register: bool = False) -> Dict[str, xr.Dataset]:
    """Read SRF data files.

    Main entry point for SRF reader. Processes one or more SRF files
    and returns dictionary of xarray datasets.

    Parameters
    ----------
    fnames : list of str or PathLike
        List of SRF file paths to read.
    metadata_only : bool, default False
        Return only metadata without loading data arrays.
    chans : list of str, optional
        Not applicable for SRF files.
    area_def : object, optional
        Not applicable for SRF files.
    self_register : bool, default False
        Not applicable for SRF files.

    Returns
    -------
    dict
        Dictionary of xarray.Dataset objects with descriptive keys.

    Notes
    -----
    Each SRF file is processed independently. Multiple files will result
    in multiple datasets in the returned dictionary.
    """
    LOG.debug(f"SRF reader called with {len(fnames)} file(s)")

    return readers.read_data_to_xarray_dict(
        fnames,
        _call_single_time,
        metadata_only,
        chans,
        area_def,
        self_register,
    )
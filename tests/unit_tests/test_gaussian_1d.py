import logging
from unittest.mock import Mock, patch

import numpy as np
import pytest
import xarray as xr

import geoips_avris_ng.plugins.modules.interpolators.gaussian_1d as gi


class TestConstants:
    """Test module constants and metadata."""

    def test_module_metadata(self):
        assert gi.interface == "interpolators"
        assert gi.family == "1d"
        assert gi.name == "1d_gaussian"

    def test_constants_values(self):
        assert gi.MIN_POINTS_REQUIRED == 2
        assert gi.DENSE_GRID_SIZE == 1000
        assert gi.GAUSSIAN_SIGMA == 2.0
        assert "x" in gi.SPATIAL_DIMS
        assert "lon" in gi.SPATIAL_DIMS
        assert "longitude" in gi.SPATIAL_DIMS
        assert "gp" in gi.AVAILABLE_METHODS
        assert "gaussian_filter" in gi.AVAILABLE_METHODS


class TestValidationFunctions:
    """Test validation functions."""

    def test_validate_method_valid(self):
        # Should not raise for valid methods
        gi.validate_method("gp")
        gi.validate_method("gaussian_filter")

    def test_validate_method_invalid(self):
        with pytest.raises(ValueError, match="Unknown method: invalid"):
            gi.validate_method("invalid")

    def test_validate_1d_data_valid(self):
        data = xr.DataArray([1, 2, 3], dims=["x"])
        gi.validate_1d_data(data)  # Should not raise

    def test_validate_1d_data_invalid(self):
        data = xr.DataArray([[1, 2], [3, 4]], dims=["x", "y"])
        with pytest.raises(ValueError, match="Only 1D data supported"):
            gi.validate_1d_data(data)

    def test_validate_sufficient_points_valid(self):
        x_vals = np.array([1, 2, 3])
        gi.validate_sufficient_points(x_vals)  # Should not raise

    def test_validate_sufficient_points_insufficient(self):
        x_vals = np.array([1])
        with pytest.raises(ValueError, match="Insufficient points"):
            gi.validate_sufficient_points(x_vals)

    def test_validate_sufficient_points_with_context(self):
        x_vals = np.array([])
        with pytest.raises(ValueError, match="after cleaning"):
            gi.validate_sufficient_points(x_vals, "after cleaning")


class TestDataTransformationFunctions:
    """Test pure data transformation functions."""

    def test_dict_to_sorted_arrays_normal(self):
        points_dict = {3.0: 30, 1.0: 10, 2.0: 20}
        x_vals, y_vals = gi.dict_to_sorted_arrays(points_dict)

        np.testing.assert_array_equal(x_vals, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(y_vals, [10, 20, 30])

    def test_dict_to_sorted_arrays_empty(self):
        x_vals, y_vals = gi.dict_to_sorted_arrays({})

        assert len(x_vals) == 0
        assert len(y_vals) == 0

    def test_dict_to_sorted_arrays_single_point(self):
        points_dict = {5.0: 25}
        x_vals, y_vals = gi.dict_to_sorted_arrays(points_dict)

        np.testing.assert_array_equal(x_vals, [5.0])
        np.testing.assert_array_equal(y_vals, [25])

    def test_arrays_to_dict(self):
        x_vals = np.array([1.0, 2.0, 3.0])
        y_vals = np.array([10, 20, 30])
        result = gi.arrays_to_dict(x_vals, y_vals)

        expected = {1.0: 10.0, 2.0: 20.0, 3.0: 30.0}
        assert result == expected

    def test_reshape_for_sklearn(self):
        arr = np.array([1, 2, 3])
        reshaped = gi.reshape_for_sklearn(arr)

        assert reshaped.shape == (3, 1)
        np.testing.assert_array_equal(reshaped.flatten(), arr)

    def test_clean_data_no_nans(self):
        x_vals = np.array([1.0, 2.0, 3.0])
        y_vals = np.array([10.0, 20.0, 30.0])
        x_clean, y_clean = gi.clean_data(x_vals, y_vals)

        np.testing.assert_array_equal(x_clean, x_vals)
        np.testing.assert_array_equal(y_clean, y_vals)

    def test_clean_data_with_nans(self):
        x_vals = np.array([1.0, np.nan, 3.0, 4.0])
        y_vals = np.array([10.0, 20.0, np.nan, 40.0])
        x_clean, y_clean = gi.clean_data(x_vals, y_vals)

        np.testing.assert_array_equal(x_clean, [1.0, 4.0])
        np.testing.assert_array_equal(y_clean, [10.0, 40.0])

    def test_clean_data_all_nans(self):
        x_vals = np.array([np.nan, np.nan])
        y_vals = np.array([np.nan, np.nan])
        x_clean, y_clean = gi.clean_data(x_vals, y_vals)

        assert len(x_clean) == 0
        assert len(y_clean) == 0


class TestInterpolationStrategies:
    """Test interpolation strategy functions."""

    @pytest.fixture
    def sample_data(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])  # x^2
        target_x = np.array([0.5, 1.5, 2.5, 3.5])
        return x, y, target_x

    @patch(
        "geoips_avris_ng.plugins.modules.interpolators.gaussian_1d.GaussianProcessRegressor",
    )
    def test_gp_interpolate(self, mock_gpr_class, sample_data):
        x_orig, y_orig, target_x = sample_data

        # Mock the GP regressor
        mock_gpr = Mock()
        mock_gpr.predict.return_value = (np.array([0.25, 2.25, 6.25, 12.25]), None)
        mock_gpr_class.return_value = mock_gpr

        result = gi.gp_interpolate(x_orig, y_orig, target_x)

        # Verify GP was configured and called correctly
        mock_gpr_class.assert_called_once()
        mock_gpr.fit.assert_called_once()
        mock_gpr.predict.assert_called_once()

        np.testing.assert_array_equal(result, [0.25, 2.25, 6.25, 12.25])

    @patch("geoips_avris_ng.plugins.modules.interpolators.gaussian_1d.interp1d")
    @patch(
        "geoips_avris_ng.plugins.modules.interpolators.gaussian_1d.gaussian_filter1d",
    )
    def test_gaussian_filter_interpolate(self, mock_filter, mock_interp, sample_data):
        x_orig, y_orig, target_x = sample_data

        # Mock interpolation functions
        mock_cubic = Mock()
        mock_cubic.return_value = np.linspace(0, 16, gi.DENSE_GRID_SIZE)

        mock_final = Mock()
        mock_final.return_value = np.array([0.25, 2.25, 6.25, 12.25])

        mock_interp.side_effect = [mock_cubic, mock_final]
        mock_filter.return_value = np.linspace(0, 16, gi.DENSE_GRID_SIZE)

        result = gi.gaussian_filter_interpolate(x_orig, y_orig, target_x)

        # Verify all components were called
        assert mock_interp.call_count == 2
        mock_filter.assert_called_once()
        np.testing.assert_array_equal(result, [0.25, 2.25, 6.25, 12.25])


class TestGaussianInterpolatePoints:
    """Test the main interpolation function."""

    @pytest.fixture
    def sample_points(self):
        return {0.0: 0.0, 1.0: 1.0, 2.0: 4.0, 3.0: 9.0}

    @pytest.fixture
    def target_coordinates(self):
        return [0.5, 1.5, 2.5]

    def test_gaussian_interpolate_points_gp_method(
        self,
        sample_points,
        target_coordinates,
    ):
        with patch(
            "geoips_avris_ng.plugins.modules.interpolators.gaussian_1d.gaussian_interpolate_points",
        ) as mock_gp:
            mock_gp.return_value = np.array([0.25, 2.25, 6.25])

            result = gi.gaussian_interpolate_points(
                sample_points,
                target_coordinates,
                method="gp",
            )

            expected = [0.25, 2.25, 6.25]
            print(result, expected)
            assert (result == expected).all()
            mock_gp.assert_called_once()

    def test_gaussian_interpolate_points_gaussian_filter_method(
        self,
        sample_points,
        target_coordinates,
    ):
        result = gi.gaussian_interpolate_points(
            sample_points,
            target_coordinates,
            method="gaussian_filter",
        )

        expected = {0.5: 0.25, 1.5: 2.25, 2.5: 6.25}
        assert isinstance(result, dict)
        rounded_result = {a: round(b, 2) for a, b in result.items()}
        assert rounded_result == expected

    def test_gaussian_interpolate_points_invalid_method(
        self,
        sample_points,
        target_coordinates,
    ):
        with pytest.raises(ValueError, match="Unknown method"):
            gi.gaussian_interpolate_points(
                sample_points,
                target_coordinates,
                method="invalid",
            )

    def test_gaussian_interpolate_points_insufficient_data(self, target_coordinates):
        points = {1.0: 1.0}  # Only one point

        with pytest.raises(ValueError, match="Insufficient points"):
            gi.gaussian_interpolate_points(points, target_coordinates)

    def test_gaussian_interpolate_points_with_nans(self, target_coordinates):
        points = {0.0: 0.0, 1.0: np.nan, 2.0: 4.0, 3.0: 9.0}

        with patch(
            "geoips_avris_ng.plugins.modules.interpolators.gaussian_1d.gaussian_interpolate_points",
        ) as mock_gp:
            mock_gp.return_value = np.array([0.25, 2.25, 6.25])

            result = gi.gaussian_interpolate_points(
                points,
                target_coordinates,
                method="gp",
            )

            # Should work with NaN values removed
            assert len(result) == 3
            mock_gp.assert_called_once()


class TestXArrayUtilityFunctions:
    """Test xarray utility functions."""

    def test_get_non_spatial_dims(self):
        data = xr.DataArray(
            np.random.rand(5, 10),
            dims=["time", "x"],
            coords={"time": range(5), "x": range(10)},
        )

        non_spatial = gi.get_non_spatial_dims(data)
        assert non_spatial == ["time"]

    def test_get_non_spatial_dims_all_spatial(self):
        data = xr.DataArray(
            np.random.rand(10),
            dims=["longitude"],
            coords={"longitude": range(10)},
        )

        non_spatial = gi.get_non_spatial_dims(data)
        assert non_spatial == []

    def test_select_array_slice_valid_index(self):
        data = xr.DataArray(
            np.random.rand(5, 10),
            dims=["time", "x"],
            coords={"time": range(5), "x": range(10)},
        )

        result = gi.select_array_slice(data, 2)

        assert "time" not in result.dims
        assert result.dims == ("x",)

    def test_select_array_slice_invalid_index(self):
        data = xr.DataArray(
            np.random.rand(3, 10),
            dims=["time", "x"],
            coords={"time": range(3), "x": range(10)},
        )

        result = gi.select_array_slice(data, 10)  # Out of range

        # Should return original array
        assert result.dims == ("time", "x")

    def test_get_target_coordinates_from_template(self):
        template = xr.Dataset(coords={"x": np.linspace(0, 10, 11)})
        fallback = np.array([1, 2, 3])

        result = gi.get_target_coordinates("x", template, fallback)

        np.testing.assert_array_equal(result, np.linspace(0, 10, 11))

    def test_get_target_coordinates_fallback(self):
        template = xr.Dataset()
        fallback = np.array([1, 2, 3])

        result = gi.get_target_coordinates("x", template, fallback)

        np.testing.assert_array_equal(result, fallback)


class TestInterpolate1DVariable:
    """Test the 1D variable interpolation function."""

    @pytest.fixture
    def input_var(self):
        return xr.DataArray(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            dims=["x"],
            coords={"x": [0, 1, 2, 3, 4]},
        )

    @pytest.fixture
    def output_template(self):
        return xr.Dataset(coords={"x": np.linspace(0, 4, 9)})

    def test_interpolate_1d_variable_success(self, input_var, output_template):
        result = gi.interpolate_1d_variable(input_var, output_template, "gp")

        assert len(result) == 9

    def test_interpolate_1d_variable_invalid_dimensions(self, output_template):
        invalid_var = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=["x", "y"],
        )

        with pytest.raises(ValueError, match="Only 1D data supported"):
            gi.interpolate_1d_variable(invalid_var, output_template, "gp")

    def test_interpolate_1d_variable_insufficient_data(self, output_template):
        sparse_var = xr.DataArray(
            [np.nan],
            dims=["x"],
            coords={"x": [0]},
        )

        result = gi.interpolate_1d_variable(sparse_var, output_template, "gp")

        assert np.all(np.isnan(result))


class TestCreateOutputDataArray:
    """Test output DataArray creation."""

    def test_create_output_dataarray_with_template_coords(self):
        data = np.array([1, 2, 3])
        template = xr.Dataset(coords={"x": np.linspace(0, 5, 3)})
        input_var = xr.DataArray([1, 2, 3], dims=["x"], coords={"x": [0, 1, 2]})
        input_var.attrs["units"] = "meters"

        result = gi.create_output_dataarray(data, template, "test_var", input_var)

        assert result.name == "test_var"
        assert result.dims == ("x",)
        assert result.attrs["units"] == "meters"
        np.testing.assert_array_equal(result.values, data)

    def test_create_output_dataarray_without_template_coords(self):
        data = np.array([1, 2, 3])
        template = xr.Dataset()
        input_var = xr.DataArray([1, 2, 3], dims=["x"], coords={"x": [0, 1, 2]})

        result = gi.create_output_dataarray(data, template, "test_var", input_var)

        np.testing.assert_array_equal(result.coords["x"].values, [0, 1, 2])


class TestProcessVariable:
    """Test variable processing function."""

    @pytest.fixture
    def input_dataset(self):
        return xr.Dataset(
            {
                "temperature": xr.DataArray(
                    [20, 21, 22],
                    dims=["x"],
                    coords={"x": [0, 1, 2]},
                ),
                "pressure": xr.DataArray(
                    [1000, 1001, 1002],
                    dims=["x"],
                    coords={"x": [0, 1, 2]},
                ),
            },
        )

    @pytest.fixture
    def output_template(self):
        return xr.Dataset(coords={"x": np.linspace(0, 2, 5)})

    def test_process_variable_success(self, input_dataset, output_template):
        # Match the mock return to template coordinate length
        coord_dim = input_dataset["temperature"].dims[0]
        expected_length = len(output_template.coords[coord_dim])

        with patch(
            "geoips_avris_ng.plugins.modules.interpolators.gaussian_1d.interpolate_1d_variable",
        ) as mock_interp:
            mock_interp.return_value = np.linspace(20, 21, expected_length)

            result = gi.process_variable(
                "temperature",
                input_dataset,
                output_template,
                None,
                "gp",
            )

            assert result is not None
            assert result.name == "temperature"
            assert len(result) == expected_length
            mock_interp.assert_called_once()

    def test_process_variable_missing_variable(self, input_dataset, output_template):
        result = gi.process_variable(
            "humidity",
            input_dataset,
            output_template,
            None,
            "gp",
        )

        assert result is None

    def test_process_variable_with_array_num(self, output_template):
        # Create multi-dimensional input
        input_dataset = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    [[20, 21, 22], [25, 26, 27]],
                    dims=["time", "x"],
                    coords={"time": [0, 1], "x": [0, 1, 2]},
                ),
            },
        )

        # Match the mock return to template coordinate length
        coord_dim = "x"  # The spatial dimension after array_num selection
        expected_length = len(output_template.coords[coord_dim])

        with patch(
            "geoips_avris_ng.plugins.modules.interpolators.gaussian_1d.interpolate_1d_variable",
        ) as mock_interp:
            mock_interp.return_value = np.linspace(25, 27, expected_length)

            result = gi.process_variable(
                "temperature",
                input_dataset,
                output_template,
                1,
                "gp",
            )

            assert result is not None
            assert len(result) == expected_length
            mock_interp.assert_called_once()


class TestMainCallFunction:
    """Test the main call function."""

    @pytest.fixture
    def input_dataset(self):
        return xr.Dataset(
            {
                "temperature": xr.DataArray(
                    [20, 21, 22, 23],
                    dims=["x"],
                    coords={"x": [0, 1, 2, 3]},
                ),
                "pressure": xr.DataArray(
                    [1000, 1001, 1002, 1003],
                    dims=["x"],
                    coords={"x": [0, 1, 2, 3]},
                ),
            },
        )

    @pytest.fixture
    def output_template(self):
        return xr.Dataset(
            {
                "temperature": xr.DataArray(
                    np.full(7, np.nan),
                    dims=["x"],
                    coords={"x": np.linspace(0, 3, 7)},
                ),
                "pressure": xr.DataArray(
                    np.full(7, np.nan),
                    dims=["x"],
                    coords={"x": np.linspace(0, 3, 7)},
                ),
            },
        )

    def test_call_function_success(self, input_dataset, output_template):
        with patch(
            "geoips_avris_ng.plugins.modules.interpolators.gaussian_1d.process_variable",
        ) as mock_process:
            mock_temp = xr.DataArray(np.linspace(20, 23, 7), dims=["x"])
            mock_press = xr.DataArray(np.linspace(1000, 1003, 7), dims=["x"])
            mock_process.side_effect = [mock_temp, mock_press]

            result = gi.call(
                area_def=None,
                input_xarray=input_dataset,
                output_xarray=output_template,
                varlist=["temperature", "pressure"],
                array_num=None,
                method="gp",
            )

            assert isinstance(result, xr.Dataset)
            assert "temperature" in result.data_vars
            assert "pressure" in result.data_vars
            assert mock_process.call_count == 2

    def test_call_function_invalid_method(self, input_dataset, output_template):
        with pytest.raises(ValueError, match="Unknown method"):
            gi.call(
                area_def=None,
                input_xarray=input_dataset,
                output_xarray=output_template,
                varlist=["temperature"],
                method="invalid",
            )

    def test_call_function_partial_success(self, input_dataset, output_template):
        with patch(
            "geoips_avris_ng.plugins.modules.interpolators.gaussian_1d.process_variable",
        ) as mock_process:
            # First variable succeeds, second fails
            mock_temp = xr.DataArray(np.linspace(20, 23, 7), dims=["x"])
            mock_process.side_effect = [mock_temp, None]

            result = gi.call(
                area_def=None,
                input_xarray=input_dataset,
                output_xarray=output_template,
                varlist=["temperature", "nonexistent"],
                method="gp",
            )

            assert "temperature" in result.data_vars
            assert (
                "nonexistent" not in result.data_vars or result["nonexistent"] is None
            )


class TestLoggingDecorator:
    """Test the logging decorator."""

    def test_log_function_call_success(self, caplog):
        @gi.log_function_call
        def dummy_function(x):
            return x * 2

        with caplog.at_level(logging.DEBUG):
            result = dummy_function(5)

        assert result == 10
        assert "Calling dummy_function" in caplog.text
        assert "Successfully completed dummy_function" in caplog.text

    def test_log_function_call_exception(self, caplog):
        @gi.log_function_call
        def failing_function():
            raise ValueError("Test error")

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ValueError, match="Test error"):
                failing_function()

        assert "Calling failing_function" in caplog.text
        assert "Error in failing_function" in caplog.text


class TestIntegration:
    """Integration tests with real data."""

    def test_end_to_end_interpolation(self):
        # Create realistic test data
        input_x = np.array([0, 2, 4, 6, 8, 10])
        input_y = np.sin(input_x) + np.random.normal(0, 0.1, len(input_x))

        input_dataset = xr.Dataset(
            {
                "signal": xr.DataArray(input_y, dims=["x"], coords={"x": input_x}),
            },
        )

        target_x = np.linspace(0, 11)
        print(input_x)
        print(input_y)
        print(target_x)
        output_template = xr.Dataset(coords={"x": target_x})

        # Test both methods
        for method in ["gp", "gaussian_filter"]:
            result = gi.call(
                area_def=None,
                input_xarray=input_dataset,
                output_xarray=output_template,
                varlist=["signal"],
                method=method,
            )

            assert "signal" in result.data_vars
            assert len(result["signal"]) == len(target_x)
            assert not np.any(np.isnan(result["signal"].values))

    @pytest.mark.parametrize("method", ["gp", "gaussian_filter"])
    def test_parameterized_methods(self, method):
        points = {0.0: 0.0, 1.0: 1.0, 2.0: 4.0, 3.0: 9.0, 4.0: 16.0}
        target_x = [0.5, 1.5, 2.5, 3.5]

        result = gi.gaussian_interpolate_points(points, target_x, method=method)

        assert len(result) == len(target_x)
        assert all(isinstance(k, float) for k in result.keys())
        assert all(isinstance(v, float) for v in result.values())


# Fixtures for test data
@pytest.fixture
def sample_points_dict():
    """Sample points dictionary for testing."""
    return {0.0: 1.0, 1.0: 2.5, 2.0: 3.2, 3.0: 2.8, 4.0: 4.1, 5.0: 5.0}


@pytest.fixture
def sample_xarray_1d():
    """Sample 1D xarray DataArray."""
    return xr.DataArray(
        [10, 15, 12, 18, 20],
        dims=["x"],
        coords={"x": [0, 1, 2, 3, 4]},
        attrs={"units": "celsius", "description": "temperature"},
    )


@pytest.fixture
def sample_dataset():
    """Sample xarray Dataset."""
    return xr.Dataset(
        {
            "temperature": (["x"], [20, 21, 22, 23, 24]),
            "pressure": (["x"], [1000, 1001, 1002, 1003, 1004]),
        },
        coords={"x": np.linspace(0, 4, 5)},
    )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

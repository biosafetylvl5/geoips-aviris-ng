# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

#!/bin/bash

# @ Identify and update all instances of @ found within this script appropriately.
# @  @package@ should be the name of this repository
# @ NOTE: imagery_clean preferred to avoid dependencies on external plotting packages
# @       for image consistency,
# @       though you may change the output format if required for your package testing.

# @ Note you may update other arguments within this test call, or add additional
# @ arguments.
# @ Also - the first time you run your test script, it will prompt you to update your
# @ test outputs - at that point you can populate the "tests/outputs" directory,
# @ and your next run will result in a 0 return value.

# @ Please create a separate test script for each piece of functionality you would
# @ like to test.

# @ Ensure each test script is called from the @package@/tests/test_all.sh script.

# @ Point to your desired test data file:
#     $GEOIPS_TESTDATA_DIR/test_data_@datatype@/data/@datafile@ \

# @ Select readername that matches the data file above:
#     --reader_name @readername@ \

# @ Select a product that is available for the above data type
#     --product_name @product@ \

# @ Point to the appropriate comparison directory, where you would like to store your
# @ test outputs. Test output directory name should match the test script name,
# @ by convention.
#     --compare_path "$GEOIPS_PACKAGES_DIR/@package@/tests/outputs/@test_single_source@" \

# @ Modify the 'geoips run single_source' command appropriately for your use case.
# @ Notes above on individual arguments.

    # @ NOTE: --resampled_read argument required for ABI and AHI readers.
    # @ Move into "geoips run single_source" call if using ABI/AHI.
    # --resampled_read \

geoips run single_source \
    $GEOIPS_TESTDATA_DIR/test_data_amsr2/data/AMSR2-MBT_v2r2_GW1_s202005180620480_e202005180759470_c202005180937100.nc \
    --reader_name amsr2_netcdf \
    --product_name "89-PCT-Using-Product-Defaults" \
    --compare_path "$GEOIPS_PACKAGES_DIR/template_basic_plugin/tests/outputs/amsr2.global_clean.89-PCT-Using-Product-Defaults" \
    --output_formatter imagery_clean \
    --filename_formatter geoips_fname \
    --minimum_coverage 0 \
    --sector_list global
ss_retval=$?

exit $((ss_retval))

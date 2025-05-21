# # # This source code is subject to the license referenced at
# # # https://github.com/NRLMMD-GEOIPS.

#!/bin/bash

currdir=`dirname $0`

geoips run config_based $GEOIPS_TESTDATA_DIR/test_data_amsr2/data/AMSR2-MBT_v2r2_GW1_s202005180620480_e202005180759470_c202005180937100.nc \
    --output_config $currdir/test_config.yaml
config_retval=$?

exit $((config_retval))

# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Add quadmix, alignment modules
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

#!/bin/bash

# sh test.sh work_dirs
 

# # gta-cnn
# TEST_ROOT=$1
# CONFIG_FILE="${TEST_ROOT}/gtaHR2csHR_udass_hrda_cnn.py"  # or .json for old configs
# CHECKPOINT_FILE="${TEST_ROOT}/udass_image_gta_cnn.pth"  # iter_36000_image_gta_cnn
# SHOW_DIR="${TEST_ROOT}/preds_gta_cnn_udass"

# # gta-trans
TEST_ROOT=$1
CONFIG_FILE="${TEST_ROOT}/gtaHR2csHR_udass_hrda_650a8.py"  # or .json for old configs
CHECKPOINT_FILE="${TEST_ROOT}/udass_image_gta_trans.pth"  # iter_49500_image_gta_trans
SHOW_DIR="${TEST_ROOT}/preds_gta_trans_udass"

# # syn-cnn

# TEST_ROOT=$1
# CONFIG_FILE="${TEST_ROOT}/synthiaHR2csHR_udass_hrda_cnn.py"  # or .json for old configs
# CHECKPOINT_FILE="${TEST_ROOT}/udass_image_syn_cnn.pth"  # iter_32500_image_syn_cnn
# SHOW_DIR="${TEST_ROOT}/preds_syn_cnn_udass"

# # syn-trans

# TEST_ROOT=$1
# CONFIG_FILE="${TEST_ROOT}/synthiaHR2csHR_udass_hrda_650a8.py"  # or .json for old configs
# CHECKPOINT_FILE="${TEST_ROOT}/udass_image_syn_trans.pth"  # iter_30500_image_syn_trans
# SHOW_DIR="${TEST_ROOT}/preds_syn_trans_udass"


echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE
echo 'Predictions Output Directory:' $SHOW_DIR
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --show-dir ${SHOW_DIR} --opacity 1


#    Copyright 2022, Intel Corporation.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

#!/bin/bash
set -x

############################################# data preprocess #############################################
cd nnUNet/nnunet

# data rename and copy
python dataset_conversion/amos_convert_label.py
python dataset_conversion/kits_convert_label.py basic

# data verify
nnUNet_plan_and_preprocess -t 507 --verify_dataset_integrity
nnUNet_plan_and_preprocess -t 508 --verify_dataset_integrity

# source data normalization
nnUNet_plan_and_preprocess -t 508 -pl2d None -pl3d ExperimentPlanner3D_v21_customTargetSpacing_kits19

# target data normalization
nnUNet_plan_and_preprocess -t 507 -pl2d None -pl3d ExperimentPlanner3D_v21_customTargetSpacing_kits19 -no_pp
python dataset_conversion/kits_convert_label.py intensity
nnUNet_plan_and_preprocess -t 507 -pl2d None -pl3d ExperimentPlanner3D_v21_customTargetSpacing_kits19 -no_plan





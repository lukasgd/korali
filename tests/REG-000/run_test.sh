#!/bin/bash

##############################################################################
# Brief: Checks for a correct installation of Korali and its modules.
# Type: Regression Test 
# Description:
# Checks whether the Korali module is correctly installed, and then checks
# the rest of its modules.
# Steps: 
# 1 - Operation: Check the existence of the korali.engine module.
#     Expected Result: The module is found, and rc = 0.
# 2 - Operation: Checking Korali's modules.
#     Expected Result: All modules execute correctly and rc = 0.
###############################################################################

###### Auxiliar Functions and Variables #########

source ../functions.sh

############# STEP 1 ##############

logEcho "[Korali] Checking Pip Installation"
pip check korali
check_result

############# STEP 2 ##############

logEcho "[Korali] Checking korali.plotter"
python3 -m korali.plotter --check
check_result

logEcho "[Korali] Checking korali.cxx"
python3 -m korali.cxx --cflags
check_result

python3 -m korali.cxx --compiler
check_result

python3 -m korali.cxx --libs
check_result

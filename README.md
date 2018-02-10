# 2018-Vision 
# To play around with python code:
Copy the contents of ArgosTargetDetection.py to Jevois Sandbox module (pythonSandbox.py) until a separate module is built
-parameters to tune
-HUE_MIN, HUE_MAX (ideal 30 to 80)
-SAT_MIN, SAT_MAX 
-VAL_MIN, VAL_MAX
-AREA_THRESHOLD
-MIN_ASPECT_RATIO_THRESHOLD


# To run the ArgosVsion module in C++:
- copy src folder contents into src folder of the newly created module.
- To create a module get the jevois base repos from: https://github.com/jevois/jevois.git and https://github.com/jevois/jevoisbase.git and https://github.com/jevois/jevois_sdk_dev.git

- Then open a terminal and do: ~/path_to_jevois_repos/jevois/scripts/create_new_module.sh Argos ArgosVision
- Now copy the src contents into newly created modules src folder (overwrite)
- In order to compile one MUST have 16.04 or 17.10 (17.04 is no longer supported from February 2018).
- IMPORTANT: before compiling follow the instruction from https://jevois.usc.edu/ and get the appropriate dependencies
- now navigate to base folder of the module and run: ./rebuild-platform.sh to compile for ARM(jevois).
- After compiling copy the jvpkg folder to packages folder of Jevois camera(microSD) and restart.
- parameters can be found in script.cfg file of the module.
- As of 02/10/2018 all parametrs in the code are made configurable and can be modified via serial commands on the fly.


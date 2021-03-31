# NHEET1D
NHEET Project 1D model

Developers:
P. Gareau;
D. Cerantola

Description:
Code solves the 1D conservation of energy equation using Schumann's Model (or variant) to obtain fluid and solid temperature distributions through a packed bed. Requires a temperature input function (sinusoidal with time currently forced) or a csv data file. The code has parametric capability.

Notes:
Mar23,2021
-program now requires 5 .py files. NHEET_main_looper has the main executable function. NHEET_jupyterlab_frontend shows a sample set of inputs required by the main function.
-NHEET_Tprofile_fns post-processes dataset for a temperature input function to provide inputs for a steady-state particle-resolved CFD analysis. Note that in cfd study, packed bed inlet is at y=L and outlet at y=0.
-some global variables are also defined as local variables; this may cause issues.
-see the individual files for a list of revisions.
Mar30,2021
-may need to add some additional functions (termcolor) with 'pip install <fn>' 
-functions updated to cooperate with Anaconda3-2020.11-Windows-x86_64.exe
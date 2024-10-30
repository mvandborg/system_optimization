# MI Investigation Project

This project is focused on conducting investigations related to Modulation instability (MI) in optical fibers, and investigating the noise properties of MI in a Brillouin Optical Time Domain Reflectometer. A sqaure optical pulse (denoted the probe) of duration T0 and peak power Ppr0 is launched into an optical fiber of length L. The script models the propagation of the pulse. The core script for running this investigation is:

- `MI_invastigation.py`

which runs a single simulation and plots the results. A sweep over the input powers Ppr0 can be run using the script 

- `MI_invastigation_sweep.py`

This script saves the results into the folder which is given as "savedir" in the code. The results can be plotted using 

- `plotting_MInoise.py`

Please note that the remaining scripts in this repository is either a bit outdated or still under construction.

Clone this repository:
   ```bash
   git clone https://github.com/mvandborg/system_optimization.git
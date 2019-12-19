# Near_Repeat_Calculator
Near repeat calculator is based on the postulate that “if a location is the target of a crime then homes within relatively short distance from the location have an increased chance of becoming a crime location within a limited number of weeks” (Ratcliffe & Rengert, 2008, p .58).
The method utilizes knox test for space-time clustering, which is based on Monte-Carlo simulations. As the method is computationally intensive for large datasets, we have developed a parallel version of near repeat calculator. A new K-time stack algorithm is developed, which stacks up randomly permuted time data for a set of simulation and performs space time calculation in a single pass, thus reducing number of pairwise distance calculation. The project utilises ForEST (domain-specific language FOR Expressing Spatial-Temporal (FOREST) computation https://github.com/eshook/Forest) as a source for the datastructures required for spatio-temporal computation. 

The following libraries are required for the near_repeat calculator to function
------------------------------------------------------------------------------------

1) binpacking (for load balancing domains) - pip install binpacking
2) numba (for fast numerical computation using JIT) - pip install numba
3) scipy.stats (for matrix rank calculation) - pip install scipy
4) numpy (for matrix operations) - pip install numpy
5) shapely (for subdomain halo calculations) - pip install shapely
6) gdal (required for shapely) - pip install gdal
7) dateutil (required for time manipulations) - pip install python-dateutil

Usage -- Run the near_repeat_calculator.py file with the following arguments

--distancebandinmeters       -   Distance band for near repeat calculator
--timebandindays             -   Time band for near repeat calculator
--maximumdistmeters          -   Maximum distance up to which to calculate near repeat patterns. Example, if distance band is 50 and maximum distance is 200, then the spatial bands will 0-50,50-100,100-150,150-200
--maxdays                    -   Maximum days up to which to calculate near repeat patterns. Example, if time band is 14 days and maximum time is 70days, then the time bands will 0-14,14-28,28-42,42-56,56-70
--kstack                     -   Time stack size for k-time stack algorithm. 
--numproc                    -   Total number of processes used for running the algorithm
--totalsim                   -   Total Monte Carlo simulations. Typical values include 99 (alpha .01), 999 (aplha .001)
--numblocks                  -   Total subodmains. Should be a power of 2
--filepath                   -   File containing the crime data (space-time event data)

Example Usage
---------------------
python near_repeat_calculator.py --distancebandinmeters 100 --timebandindays 14 --maximumdistmeters 1000 --maxdays 70 --kstack 10 --numproc 8 --totalsim 99 --numblocks 256 --filepath crime100.csv

The program generates and prints the knox table of p-values and the knox ratio table

The spatio-temporal event data should be strictly in this format

id,t,y,x
10503671,4/28/2016 23:40,42.02253659,-87.67374743

Every point should have a unique ID and time and geographical coordinates



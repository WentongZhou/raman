# Automate raman-analyzer 
The long-term goal of this script is to automate peak-fitting process of Raman Spectrum.\
The typical procedure to process raman spectrum would include:
- Import the data
- Smooth the line
- Baseline reduction
- Peak and Shoulder Peak finder
- Peak Fitting 

## Dependencies
The script is based on `numpy`,`scipy`,`rampy`,`matplotlib`,`lmfit` to do 
data analysis and visualization.

## Simple Usage
After running the codes, call \
`s = raman_analyzer(name,min,max,filter)`
- name: your Raman datafile name with full directory (string)
- min,max : the range of wavenumbers you are interested in analyzing
- filter: the filtration threshold for Raman peaks points\
(the more the filter number is , the less peak points would be stored).
- `s` would be the object in which all the information is stored. 
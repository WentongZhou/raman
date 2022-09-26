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

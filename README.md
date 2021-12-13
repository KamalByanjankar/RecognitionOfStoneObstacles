# Recognition Of Stone Obstacles
Recognition of small stone obstacles for Autonomous Driving using Red pitaya and SRF02 ultrasonic sensor.

# Requirements
The packages required are:
* numpy 
* pandas
* matplotlib
* glob
* sklearn

# Data Acquisition
For the data acquisition process, the red pitaya is connected to the pc using ip address. Once, the connection between red pitaya and the pc is successful, then we can capture data of an object and save the file with .csv extension.

# Data Analysis
The captured data consists of data and other unwanted factors which were added in the data during data acquisition. So, first the captured data is loaded, and the offset voltage is removed from each data frame, which aligns the plot with x-axis. The threshold voltage is determined using an empty space and found to be 0.003V. The captured data consist of both transmitted and reflected or echo signals. So, the first 4000 data points are removed because we are only interested in the reflected or echo signals.

# Butterworth Bandpass Filter
The captured signals consist of noise signals also which are added during data acquisition process. So, to remove the noise signals from data, Butterworth band pass filter is used. Since the operating frequency of an ultrasonic sensor is 40 KHz, the reflected signals should lie near 40 KHz. So, the signals above 30 KHz and below 50 KHz are only considered which helps in detection of echo signals. 

# Echo Separation
Rolling window is used to identify the exact echo signal that we require for further process. It is done by sliding the rolling window and capturing the peak values and its index. Once the peak value along with its index is found, then the echo signals that we require is captured. The echo signals are saved in a separate file with .csv extension.

# FFT Implementation
The echo signals are time domain data, which are converted into frequency domain data using FFT. The frequency range of 30 KHz to 50 KHz is applied in FFT for conversion process, which provides features and are saved in a separate file with .csv extension. 

# Machine Learning
The extracted features using FFT is used to train Machine Learning models. Before training the ML models, the data is splitted using train_test_split library imported from sklearn. 



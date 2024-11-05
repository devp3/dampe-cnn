# Convolutional Neural Network for trajectory regression

## Setup
Dark Matter Particle Explorer (DAMPE) is a space based high-energy cosmic and gamma ray observatory. The detector measures high-energy spectrum of cosmic electrons and gamma rays. One of the sub-detectors of DAMPE is a calorimeter constructed from Bismuth Germanate Oxide Crystal read out using PMTs. 

The BGO calorimeter PMTs are arranged in an "X-Y" (criss-cross) fashion to localize the position of a high-energy particle passing through and depositing energy in the calorimeter. The calorimeter consists of 22 bars, each bar is 25 mm x 25 mm x 600 mm in dimensions. 

The zipped data files are stored in [data.zip](https://github.com/devp3/dampe-cnn/blob/main/data.zip).

## Challenge
The goal of this project is to infer the x-y co-ordinates of the incident particle at the top and bottom of the calorimeter using the images of particle showers. Besides the images, the data set consists of scalar values which are the maximum deposited energy and the maximum energy deposited in one PMT bar. The images are hits in the calorimeter, essentially an image of the particle shower. Each image is of dimension 14 (layers) x 22 (bars) x 1 (channel). The channel value is the pixel density between [0, 255].

The dataset contains 141946 images which are split into 70% training, 20% testing, and 10% validation. The images and energies are processed using `sklearn.prerocessing.MinMaxScaler`. 

## Network
Convolutional Neural Network (CNN) is set up to process the image and the energy values to regress the x-y co-ordinates of the particle entry and exit in the calorimeter. The CNN model is shown [here](https://github.com/devp3/dampe-cnn/blob/main/model.png). The network was built using Keras and was trained using the Adagrad algorithm with the Mean Squared Error (MSE) as the loss function. The CNN was trained for 500 epochs with an EarlyStopping after 50 epochs.

## Results

The notebook `Evaluation.ipynb` contains several plots showing the predicted versus the truth x-y co-ordinates of the incident particles. The network response is linear, as shown in the [linearity plot](https://github.com/devp3/dampe-cnn/blob/main/plots/pred_linearity.pdf). The predicted co-ordinates have a resolution of 12.3 mm in both the x- and y-directions as shown in the [bias plot](https://github.com/devp3/dampe-cnn/blob/main/plots/fit_bias.pdf).
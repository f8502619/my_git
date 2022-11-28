#Objective: The objective is to produce a deep convolutional neural network (DCNN) model and a evaluation metric for image segmentation. 

#Architecture: The model is based on the U-Net architecture.The U-Net is a very popular architecture for biomedical image segmentation, where its encoder-decoder structure 
is very suitable for this task. It initialy downsamples the input image and then upsamples it to the original size. The downsampling is done by a series of convolutional layers followed 
by max pooling layers. The upsampling is done by a series of convolutional layers followed by upsampling layers. The encoder part of the network is used to extract features from the
input image, while the decoder part of the network is used to upsample the features to the original size. The encoder part of the network is also used to extract features from the 
output of the upsampling layers. These features are then concatenated with the features from the encoder part of the network and then passed through the upsampling layers.
The final output of the network is a segmentation mask of the input image. The segmentation mask is a binary image where each pixel is assigned a class label. 
The class labels are 0 for background and 1 for foreground.

#Mode
eval: 0-turn off for training; 1-turn on for directly loading the pretrained model

# Dataset 
Image data: dataset/train_images_256/
mask data: dataset/train_masks_256/

#Run and environment settings
To run the code, simply run the shell script ./train.sh in the terminal with the prerequisites installed. All the prerequisities are mentioned in the requirement.txt file. 
The shell script will run the main.py file. The final predictions are made across the entire dataset, with the output saved as .tif files in the dataset/experiments/exp_1/model/pred/  folder,
just as with the masks provided. The trained model is saved in the model/ directory with the name best_model.pth

# Loss function
The loss function chosen for this task is the cross entropy loss. The reason this loss function is chosen is because the minimisation of the cross entropy corresponds to the maximum 
likelihood estimation of the network parameters with respect to the likelihood distribution of the decision given the dataset. This is assuming the decisions are independent.

#Training settings
Training was done on batch size 16, using the Adam optimiser with learning rate 0.0003. The maximum epoch was set at 1000, with early stopping to prevent overfitting. 
The early stopping patience was 10 epochs, with minimum loss improvement 0.001 required. The best model is saved in the model/ folder. 
These values can all be modified in the train.sh script. From my own training, the final prediction cross entropy loss was found to be 0.083.
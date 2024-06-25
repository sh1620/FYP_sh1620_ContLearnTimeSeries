# FYP_sh1620_ContLearnTimeSeries
GitHub repository of the code deployed in my Final Year Project for Contrastive Learning on Time Series Data

# How to use
Each model is in a .ipynb file for use in google collab. With the proper directories and arguments set, the code should work from just running the entire notebook.

Args: a set of args used in the project will be in a .txt file, however they can be tweaked inside each google collab file if you want to change them.

Notable args to change include: selected dataset, batch_size for training, sqrt(size of sample in dataset) for use in a NxN image, number of classes in a dataset

# TS-TCC

Note the data_preprocessing scripts in the folder are used to generate files essential for all models.

To deploy the TS-TCC model, there is a Configs class where recommended configs are in the config_files folder. THe notable ones are self.features_len, self.batch_size and self.num_classes which may require particular tweaking based on the dataset's properties. In the args section, training_mode should be set to self_supervised for training and train_linear for linear probing to evaluate the models. Note that this will require running the code twice. selected_dataset in args should be changed where needed to select the dataset.

# SimCLR

For SimCLR, note that this file is responsible for generating the files to convert the time series data to images and is therefore required in both SimCLR and MoCo. Note the data prep cell is optional when the data is already generated.

When operating this the required changeable parameters are in class SimCLRPneumoniaMNISTDataset(), self.size is the NxN size of the data, npz_file selects the data too. It also requires changes for linear probing, firstly the changes as before sith self.size and npz_file need altering per dataset, with the line "classifier = LinearProbeClassifier(model, num_classes=5)" requiring changing the num_classes section to reflect the number of classes in the selected dataset. Finally batch_size may need changing as appropriate.

# MoCo

Both training and linear probing have their own args section which outlines the needed changes. Those being data, size, batch_size, and num_classes for linear probing. The rest only need to be changed once for the environment being deployed so should work elsewhere.

# Other use notes

For use in google collab:
Make appropriate changes to parent directories within google drive, check args and run.

For use outside:
May require changes to remove the two lines:
"from google.colab import drive
drive.mount('/content/drive')"
as appropriate from the start of the files, or comment them out. Will still require changes to directory names.

To train the generic supervised classifier, comment out the lines:
        "# Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False"
Within the LinearProbeClassifier class.

# Note
The original Electronic Motor Dataset files are too large, even when compressed, to be uploaded to GitHub though the processed files are present and have been uploaded

# FYP_sh1620_ContLearnTimeSeries
GitHub repository of the code deployed in my Final Year Project for Contrastive Learning on Time Series Data

# How to use
Each model is in a .ipynb file for use in google collab. With the proper directories and arguments set, the code should work from just running the entire notebook.

Args: a set of args used in the project will be in a .txt file, however they can be tweaked inside each google collab file if you want to change them.
Notable args to change include: selected dataset, batch_size for training, sqrt(size of sample in dataset) for use in a NxN image, number of classes in a dataset
TS-TCC only: features_len, training_mode
For these args, some recommendations will be in comments rather than just the .txt files.

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

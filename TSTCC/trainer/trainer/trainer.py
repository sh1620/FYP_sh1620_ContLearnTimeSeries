import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('TSTCC Confusion Matrix')
    plt.savefig('confusionmatrix.png')
    plt.close()  # Close the figure

def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_dl, config, device, training_mode)
        valid_loss, valid_acc, valid_f1_score, valid_recall, valid_precision, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        if training_mode != 'self_supervised':  # use scheduler in all other modes.
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, test_f1_score, test_recall,test_precision, _, _  = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')
        logger.debug(f'Test F1 Score     :{test_f1_score:0.4f}\t | Test Recall     : {test_recall:0.4f} | Test Precision     : {test_precision:0.4f}')

        # Save test results to output.txt file
        output_file = os.path.join(experiment_log_dir, "output.txt")
        result_array = [test_acc, test_f1_score, test_recall, test_precision]
        with open(output_file, 'w') as f:
            for value in result_array:
                f.write(f'{value:.4f}\n')

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config, device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    # store features and labels for t-SNE
    features_store = []
    labels_store = []

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised":
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)

            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)

            # normalize projection feature vectors
            zis = temp_cont_lstm_feat1 
            zjs = temp_cont_lstm_feat2 


        else:
            output = model(data)

        # compute loss
        if training_mode == "self_supervised":
            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(zis, zjs) * lambda2
            
        else: # supervised training or fine tuining
            predictions, features = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())


        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()

    return total_loss, total_acc




def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                output = model(data)

            # compute loss
            if training_mode != "self_supervised":
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())


                ## Compute t-SNE
                #tsne = TSNE(n_components=2, random_state=0)
                ##transformed_features = tsne.fit_transform(features.detach().cpu().numpy())
                #features_2d = features.view(features.shape[0], -1)
                #transformed_features = tsne.fit_transform(features_2d.detach().cpu().numpy())


                ## Plot t-SNE
                #plt.figure(figsize=(10,10))
                #plt.scatter(transformed_features[:, 0], transformed_features[:, 1], c=labels.detach().numpy())
                #plt.scatter(transformed_features[:, 0], transformed_features[:, 1], c=labels.cpu().detach().numpy())
                #plt.title('2D t-SNE plot of feature representations')
                #plt.savefig('TSTCC_TSNE.png')

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())


    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0

    if training_mode == "self_supervised":
        total_acc = 0
        f1 = 0
        recall = 0
        precision = 0
        return total_loss, total_acc, f1, recall, precision, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc

        #evaluation for anomaly detection
        #f1 = f1_score(trgs, outs, average='binary', pos_label=1)
        #recall = recall_score(trgs, outs, average='binary', pos_label=1)
        #precision = precision_score(trgs, outs, average='binary', pos_label=1)


        ####################additions for clustering
        # predictions, features = output
        # features_2d = features.view(features.shape[0], -1)
        # # Ensure the tensor is on the cpu and convert it to numpy
        # features_2d_np = features_2d.cpu().detach().numpy()
        # # Define the k-means model (assume k=3 for this example)
        # kmeans = KMeans(n_clusters=4)

        # # Fit the model
        # kmeans.fit(features_2d_np)

        # # Now the variable 'kmeans' is a trained model you can use to predict clusters
        # clusters = kmeans.predict(features_2d_np)

        # # Create a mapping between clusters and true classes
        # cluster_class_mapping = {}
        # for cluster_id in np.unique(clusters):
        #     class_ids_in_cluster = labels.cpu().numpy()[clusters == cluster_id]
        #     most_common_class_id = np.bincount(class_ids_in_cluster).argmax()
        #     cluster_class_mapping[cluster_id] = most_common_class_id

        # # Map the clusters to class labels
        # cluster_class_preds = np.vectorize(cluster_class_mapping.get)(clusters)

        # # Calculate metrics using these new predictions
        # f1 = f1_score(labels.cpu().numpy(), cluster_class_preds, average='macro')
        # recall = recall_score(labels.cpu().numpy(), cluster_class_preds, average='macro')
        # precision = precision_score(labels.cpu().numpy(), cluster_class_preds, average='macro')



        ########################################
        f1 = f1_score(trgs, outs, average='macro')
        recall = recall_score(trgs, outs, average='macro')
        precision = precision_score(trgs, outs, average='macro')


        # Compute class counts
        classes = np.unique(np.concatenate((outs, trgs)))
        prediction_counts = np.bincount(outs.astype(int), minlength=len(classes))
        true_counts = np.bincount(trgs.astype(int), minlength=len(classes))

        # # Plot the bar chart
        # width = 0.35
        # x = np.arange(len(classes))
        # fig, ax = plt.subplots()
        # rects1 = ax.bar(x - width/2, prediction_counts, width, label='Predicted')
        # rects2 = ax.bar(x + width/2, true_counts, width, label='True')
        # ax.set_xlabel('Class')
        # ax.set_ylabel('Count')
        # ax.set_title('Class Counts')
        # ax.set_xticks(x)
        # ax.set_xticklabels(classes)
        # ax.legend()
        # plt.savefig('bar_plot.png')
        # plt.close()

        plot_confusion_matrix(trgs, outs, classes=np.unique(trgs))
        #plot_confusion_matrix(labels.cpu().numpy(),cluster_class_preds, classes=np.unique(labels.cpu().numpy()))
        plt.title('TSTCC Confusion Matrix')
        plt.savefig('confusionmatrix.png')   

        return total_loss, total_acc, f1, recall, precision, outs, trgs


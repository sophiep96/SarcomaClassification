from __future__ import print_function, division
import matplotlib.pyplot as plt 
import sklearn.metrics
from matplotlib.pyplot import figure
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.nn.Module.dump_patches = True
# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "/home/ucabepe/Scratch/cancer_data_test" 

phase = "test"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "inception"

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for
num_epochs = 4

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer parameters
feature_extract = True

input_size = 299

y_score = []
y_true = []
y_classes = []
probs = []
tiles_per_slide = collections.defaultdict(int)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([  
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [1, 1, 1])
    ]),
    'val': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [1, 1, 1])
    ]),
    'test': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
#         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0, 0 , 0], [1, 1, 1])
    ]),
}

# customised dataloader to also load the name of the image file.
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def prepare_dataloaders(data_dir, phases, data_tranforms):
    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x),
                                      data_transforms[x]) 
                      for x in phases}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4) 
                   for x in phases}
    dataset_sizes = {x: len(image_datasets[x]) for x in phases}
    return dataloaders,image_datasets


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def extract_pathology_number(file_name):
    return file_name.split('/')[-1].split('-')[0]


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0
    if model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    return model_ft, input_size



def createOptimizer(model_ft):
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    return optimizer_ft




def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=True):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels,_ in dataloaders[phase]:
                torch.cuda.empty_cache()
                if len(inputs) == 1:
                    continue
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    # For each training element get the prediction with the highest probability.
                    # torch.Max returns an array of maximum values and the indexes of those values.
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def compare_labels_and_preds(preds, values, original_labels, files, slide_info, dataloaders):
    SFT_idx = dataloaders['test'].dataset.class_to_idx['SFT']
    SS_idx = dataloaders['test'].dataset.class_to_idx['SS']
    correct_SFT = all_SFT = 0
    correct_SS = all_SS = 0
    unknown = unknown_ss = unknown_sft = 0
    for index, pred in enumerate(preds):
        slide_number = extract_pathology_number(files[index])
        if slide_number not in slide_info:
            # Slide number => correct_pred, all_tiles, unknown_tiles, orginal_label
            slide_info[slide_number].extend([0,0,0,original_labels[index]])
        slide_info[slide_number][1] += 1
        if values[index] > 0.96:
            y_true.append(int(original_labels[index].cpu().numpy()))
            y_score.append(ss_probs[index])
            y_classes.append(int(pred.cpu().numpy()))
           
            if original_labels[index] == SFT_idx:
                all_SFT += 1
                if pred == SFT_idx:
                    correct_SFT += 1
                    slide_info[slide_number][0] += 1
            elif original_labels[index] == SS_idx:
                all_SS += 1
                if pred == SS_idx:
                    correct_SS += 1
                    slide_info[slide_number][0] += 1
        else:
            slide_info[slide_number][2] += 1
            if original_labels[index] == SS_idx:
                unknown_ss += 1
                all_SS += 1
            else:
                unknown_sft += 1
                all_SFT += 1
            unknown += 1
    return correct_SFT, all_SFT,correct_SS,all_SS, unknown, unknown_ss, unknown_sft

def calculate_accuracy_per_slide(slide_info):
    SFT_correct_slides = all_SFT_slides = 0
    SS_correct_slides = all_SS_slides = 0
    unknown_SS_slides = unknown_SFT_slides = 0
    slide_to_accuracy = {}

    for slide_number, info in slide_info.items():
        correct_pred, all_seen, unknown, tumor_type = info
        print("slide number: {}, number of tiles: {}".format(slide_number, all_seen))
        tiles_with_predictions = all_seen-unknown
        if tiles_with_predictions == 0:
            print("slide number: ", slide_number, "type: ", tumor_type, " predicted as UNKNOWN")
            slide_to_accuracy[slide_number] = -1
            if tumor_type == tumor_types['SS']:
                all_SS_slides += 1
                unknown_SS_slides += 1
            else:
                all_SFT_slides += 1
                unknown_SFT_slides += 1
            continue
        probability = correct_pred/tiles_with_predictions
        print("The probability for correct prediction: {}".format(probability))
        slide_to_accuracy[slide_number] = probability

        if tumor_type == tumor_types['SFT']:
            all_SFT_slides += 1
            if probability > 0.500:
                SFT_correct_slides += 1
        elif tumor_type == tumor_types['SS']:
            all_SS_slides += 1
            if probability > 0.500:
                SS_correct_slides += 1
    print("{}/{} SFT and {}/{} SS correctly predicted".format(SFT_correct_slides, all_SFT_slides,
                                                              SS_correct_slides, all_SS_slides))
    print("{}/{} SFT and {}/{} SS unknown slides".format(unknown_SFT_slides, all_SFT_slides,
                                                              unknown_SS_slides, all_SS_slides))
    return slide_to_accuracy
    
ss_probs = []
def predict_labels(model_ft, dataloaders):
    since = time.time()
    running_corrects = 0
    SFT_acc = SS_acc = 0
    correct_SFT = all_SFT = 0
    correct_SS = all_SS = 0
    all_unknown = unknown_SS = unknown_SFT = 0
    slide_info = collections.defaultdict(list)
    for inputs, labels, files in dataloaders[phase]:
        if len(inputs) == 1:
            continue
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model_ft(inputs)
        softmax = nn.Softmax()
        softmax_outputs = softmax(outputs)
        values, preds = torch.max(softmax_outputs, 1)
        ss_probs.extend(softmax_outputs.detach().cpu().numpy()[:,1])
        probs.extend(values.detach().cpu().numpy())
        correct_sft, all_sft, correct_ss, all_ss, unknown, unknown_ss, unknown_sft = compare_labels_and_preds(preds,values, labels.data, files, slide_info, dataloaders)
        correct_SFT += correct_sft
        all_SFT += all_sft
        correct_SS += correct_ss
        all_SS += all_ss
        all_unknown += unknown
        unknown_SS += unknown_ss
        unknown_SFT += unknown_sft
    print("number of correct_SFT: {} and correct_SS: {}".format(correct_SFT, correct_SS))
    print("number of incorrect_SFT: {} and incorrect_SS: {}".format(all_SFT-(correct_SFT+unknown_SFT), all_SS-(unknown_SS+correct_SS)))
    print("number of all_SFT: {} and all_SS: {}".format(all_SFT, all_SS))
    print("number of unknown_SFT: {} and unknown_SS: {}".format(unknown_SFT, unknown_SS))
    SFT_acc = correct_SFT/(all_SFT-unknown_SFT)
    SS_acc = correct_SS/(all_SS-unknown_SS)
    tile_acc = (correct_SFT + correct_SS)/(all_SFT+all_SS-all_unknown)
    time_elapsed = time.time() - since
    print("The accuracy among tiles are: {}".format(tile_acc))
    print("The accuracy per tile for SFT: {} and for SS: {}".format(SFT_acc, SS_acc))
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return slide_info

def create_prob_distribution(probs):
    distribution = collections.defaultdict(int)
    distribution["0_70"] = 0
    distribution["70_75"] = 0
    distribution["75_80"] = 0
    distribution["80_85"] = 0
    distribution["85_90"] = 0
    distribution["90_95"] = 0
    distribution["95_100"] = 0
    for prob in probs:
        if prob < 0.70:
            distribution["0_70"] += 1
        if prob >= 0.70 and prob < 0.75:
            distribution["70_75"] += 1
        if prob >= 0.75 and prob < 0.80:
            distribution["75_80"] += 1
        if prob >= 0.80 and prob < 0.85:
            distribution["80_85"] += 1
        if prob >= 0.85 and prob < 0.90:
            distribution["85_90"] += 1
        if prob >= 0.90 and prob < 0.95:
            distribution["90_95"] += 1
        if prob >= 0.95 and prob <= 1:
            distribution["95_100"] += 1
    return distribution

if phase == 'test':
    dataloaders, image_datasets = prepare_dataloaders(data_dir,['test'], data_transforms)
    tumor_types = image_datasets['test'].class_to_idx
else:
    dataloaders, image_datasets = prepare_dataloaders(data_dir,['train', 'val'], data_transforms)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
# Setup the loss fxn
criterion = nn.CrossEntropyLoss()
optimizer_ft = createOptimizer(model_ft)


# Train and evaluate
if phase == "test":
    model_ft = torch.load("/home/ucabepe/Scratch/models/inception2_model.pt")
    model_ft.eval()
    slides_info = predict_labels(model_ft, dataloaders)
    slide_to_accuracy = calculate_accuracy_per_slide(slides_info)
    fpr,tpr,_=sklearn.metrics.roc_curve(y_true, y_score, pos_label=1)
    p,r,f1,s = sklearn.metrics.precision_recall_fscore_support(y_true, y_classes)
    auc = sklearn.metrics.roc_auc_score(y_true, y_score)

    print("precision: {}".format(p))
    print("recall: {}".format(r))
    print("f1: {}".format(f1))
    print("support: {}".format(s))
    print("fpr: {}".format(fpr))
    print("tpr: {}".format(tpr))

    figure(num=None, figsize=(6, 5), dpi=80, facecolor='w', edgecolor='k')

    plt.plot(fpr, tpr, label="AUC: {}".format(auc))

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=1)
    plt.savefig('/home/ucabepe/Scratch/models/roc_{}_all.png'.format(model_name))

    with open('/home/ucabepe/Scratch/models/accuracy.txt', 'w') as f:
        print("per slide accuracy: ",slide_to_accuracy,file=f)  # Python 3.x
else:
    model_ft, hist = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
# Save the model
    torch.save(model_ft, "/home/ucabepe/Scratch/models/inception2_model.pt")
    with open('/home/ucabepe/Scratch/models/history.txt', 'w') as f:
        print("model_ft: ",model_ft, "history: ",hist, file=f)  # Python 3.x

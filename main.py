
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from collections import OrderedDict
import random

import os
import argparse
import numpy as np

from models import *
from models import discriminator
from utils import progress_bar
from utils import get_model
from loss import *
from torch.utils.tensorboard import SummaryWriter
from utils import create_imbalanced_dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from evaluation_dataset import testloader
from tqdm import tqdm
from utils_for_evaluation import calculate_acc, calculate_scores



# ================= Arugments ================ #

parser = argparse.ArgumentParser(description='PyTorch CASIA Training')
parser.add_argument('--run', default='2', type=str, help='experiment number')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--d_lr', default=1e-3, type=float, help='discriminator learning rate')
# teachers
parser.add_argument('--teachers', default='[\'resnet34\',\'resnet50\']', type=str, help='teacher networks type')
parser.add_argument('--student', default='resnet18', type=str, help='student network type')
parser.add_argument('--dataset', default='casia', type=str, help='dataset type')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--gamma', default='[1,1,1,1,1]', type=str, help='')
parser.add_argument('--eta', default='[1,1,1,1,1]', type=str, help='')
parser.add_argument('--fc_out', default=1, type=int, help='if immediate output from fc-layer')
parser.add_argument('--loss', default="ce", type=str, help='loss selection')
parser.add_argument('--adv', default=1, type=int, help='add discriminator or not')
parser.add_argument('--name', default='casia_contrastive', type=str, help='the name of this experiment')
parser.add_argument('--pool_out', default="avg", type=str, help='the type of pooling layer of output')
parser.add_argument('--out_layer', default="[-1]", type=str, help='the type of pooling layer of output')
parser.add_argument('--out_dims', default="[10000,5000,1000,500,10572]", type=str, help='the dims of output pooling layers')
parser.add_argument('--teacher_eval', default=0, type=int, help='use teacher.eval() or not')

# model config
parser.add_argument('--depth', type=int, default=26)
parser.add_argument('--base_channels', type=int, default=96)
parser.add_argument('--grl', type=bool, default=False, help="gradient reverse layer")

# run config
parser.add_argument('--outdir', type=str, default="results")
parser.add_argument('--seed', type=int, default=17)
parser.add_argument('--num_workers', type=int, default=7)

# optim config
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--base_lr', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nesterov', type=bool, default=True)
parser.add_argument('--lr_min', type=float, default=0)


args = parser.parse_args()

# ================= Config Collection ================ #

model_config = OrderedDict([
    ('depth', args.depth),
    ('base_channels', args.base_channels),
    ('input_shape', (1, 3, 112, 112)),
    ('n_classes', 10572),
    ('out_dims', args.out_dims),
    ('fc_out', args.fc_out),
    ('pool_out', args.pool_out)
])

optim_config = OrderedDict([
    ('epochs', args.epochs),
    ('batch_size', args.batch_size),
    ('base_lr', args.base_lr),
    ('weight_decay', args.weight_decay),
    ('momentum', args.momentum),
    ('nesterov', args.nesterov),
    ('lr_min', args.lr_min),
])

data_config = OrderedDict([
    ('dataset', 'CASIA'),
])

run_config = OrderedDict([
    ('seed', args.seed),
    ('outdir', args.outdir),
    ('num_workers', args.num_workers),
])

config = OrderedDict([
    ('model_config', model_config),
    ('optim_config', optim_config),
    ('data_config', data_config),
    ('run_config', run_config),
])

print(args)

# ================= Initialization ================ #

os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_id
device = 'cuda'
# device = 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
#changes for face
best_accuracy =0.0

# ================= Data Loader ================ #

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Changes for face
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.5162327, 0.39947282, 0.34154985], std=[0.2852052, 0.24476086, 0.23453709])  # Normalize with ImageNet means and stds
])


if args.dataset=='cifar100':
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
elif args.dataset=='cifar100_imbalanced':
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainset_imbalanced = create_imbalanced_dataset(trainset)
    trainloader = torch.utils.data.DataLoader(trainset_imbalanced, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
elif args.dataset=='casia':
    # Load the dataset from the folder "dataset"
    dataset = datasets.ImageFolder(root='./dataset', transform=transform)
    # Create a DataLoader to load data in batches
    trainloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)


optim_config['steps_per_epoch'] = len(trainloader)

# ================= Model Setup ================ #

args.teachers = eval(args.teachers)

print('==> Training', args.student if args.name is None else args.name)
print('==> Building model..')

# get models as teachers and students
teachers, student = get_model(args, config, device="cuda")

print("==> Teacher(s): ", " ".join([teacher.__name__ for teacher in teachers]))
print("==> Student: ", args.student)

dims = [student.out_dims[i] for i in eval(args.out_layer)]
print("dims:", dims)

update_parameters = [{'params': student.parameters()}]

if args.adv:
    discriminators = discriminator.Discriminators(dims, grl=args.grl)
    for d in discriminators.discriminators:
        d = d.to(device)
        if device == "cudr":
            d = torch.nn.DataParallel(d)
        update_parameters.append({'params': d.parameters(), "lr": args.d_lr})

print(args)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/%s-generator/ckpt.t7' % "_".join(args.teachers))
    student.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']

# ================= Loss Function for Generator ================ #

if args.loss == "l1":
    loss = F.l1_loss
elif args.loss == "l2":
    loss = F.mse_loss
elif args.loss == "l1_soft":
    loss = L1_soft
elif args.loss == "l2_soft":
    loss = L2_soft
elif args.loss == "ce":
    loss = CrossEntropy      # CrossEntropy for multiple classification

criterion = betweenLoss(eval(args.gamma), loss=loss)

# ================= Loss Function for Discriminator ================ #

if args.adv:
    discriminators_criterion = discriminatorLoss(discriminators, eval(args.eta))
else:
    discriminators_criterion = discriminatorFakeLoss()

# ================ Contrastive Loss Function =================== #

contrastive_criterion = ContrastiveLoss(batch_size=args.batch_size)
contrastive_criterion_test = ContrastiveLoss(batch_size=100)

# ================= Optimizer Setup ================ #

if args.student == "densenet_cifar":
    optimizer = optim.SGD(update_parameters, lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150 * min(2, len(teachers)), 250 * min(2, (len(teachers)))],gamma=0.1)
    print("nesterov = True")
elif args.student == "mobilenet":
    optimizer = optim.SGD(update_parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)  # nesterov = True, weight_decay = 1e-4，stage = 3, batch_size = 64
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150 * min(1, len(teachers)), 250 * min(1, (len(teachers)))],gamma=0.1)
else:
    optimizer = optim.SGD(update_parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)  # nesterov = True, weight_decay = 1e-4，stage = 3, batch_size = 64
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150 * min(1, len(teachers)), 250 * min(1, (len(teachers)))],gamma=0.1)


# ===================== Initializing Tensorboard ======================== #

writer = SummaryWriter(log_dir='logs/run' + args.run)

# ================= Training and Testing ================ #

def teacher_selector(teachers):
    idx = np.random.randint(len(teachers))
    return teachers[idx]

def output_selector(outputs, answers, idx):
    return [outputs[i] for i in idx], [answers]

def train(epoch):
    print('\nEpoch: %d' % epoch)
    scheduler.step()
    student.train()
    train_loss = 0
    correct = 0
    total = 0
    discriminator_loss = 0
    crossentropy_loss = 0
    contrastive_loss = 0
    # changes for face
    global best_accuracy

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        num_batches = len(trainloader)
        total += targets.size(0)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Get output from student model
        outputs = student(inputs)
        # Get teacher model
        teacher = teacher_selector(teachers)
        # Get output from teacher model
        answers = teacher(inputs)
        # Select output from student and teacher
        outputs, answers = output_selector(outputs, answers, eval(args.out_layer))
        # Calculate similarity loss between student and teacher
        loss = criterion(outputs, answers)
        lenght = len(outputs)
        cntrstv_loss = sum(contrastive_criterion(outputs[i], answers[i]) for i in range(lenght))

        total_loss = loss + cntrstv_loss


        total_loss.backward()
        optimizer.step()

        train_loss += loss.item()
        contrastive_loss += cntrstv_loss.item()
        _, predicted = outputs[-1].max(1)
        correct += predicted.eq(targets).sum().item()


        # Progrerss bar for contrastive loss + similarity loss
        progress_bar(batch_idx, len(trainloader), 'Teacher: %s | Lr: %.4e | Similarity_Loss: %.3f | Contrastive_Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (teacher.__name__, scheduler.get_lr()[0], train_loss / (batch_idx + 1), contrastive_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # add log of the training to the writer
        writer.add_scalar('Accuracy/Train', 100. * correct / total, batch_idx*(epoch+1))
        writer.add_scalar('Similarity_loss/Train', train_loss / (batch_idx + 1), batch_idx*(epoch+1))
        writer.add_scalar('Learning_rate', scheduler.get_lr()[0], batch_idx*(epoch+1))
        writer.add_scalar('Contrastive_loss/Train', contrastive_loss / (batch_idx + 1), batch_idx*(epoch+1))

        # changes for face
        # Validation loop
        if (batch_idx % 3000 == 1):
            # Validation
            dist_list = torch.empty(0)
            dist_list = dist_list.to(device)
            labels_list = torch.empty(0)
            labels_list = labels_list.to(device)

            preds = []
            labels = []

            student.eval()

            with torch.no_grad():
                for data in tqdm(testloader):
                    image1, image2, img1_h, img2_h, pair_label = data
                    image1 = image1.to(device)
                    image2 = image2.to(device)
                    img1_h = img1_h.to(device)
                    img2_h = img2_h.to(device)
                    pair_label = pair_label.to(device)

                    embedding1 = student(image1)[-1]
                    embedding2 = student(image2)[-1]
                    embedding1_h = student(img1_h)[-1]
                    embedding2_h = student(img2_h)[-1]

                    feature1 = torch.cat((embedding1, embedding1_h), dim=1)
                    feature2 = torch.cat((embedding2, embedding2_h), dim=1)

                    # distance based on cosine similarity
                    cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
                    pred = cosine_sim(feature1, feature2)

                    preds += pred.data.cpu().tolist()
                    labels += pair_label.data.cpu().tolist()

                # Calculate accuracy and scores
                accuracy, _ = calculate_acc(preds, labels)
                calculate_scores(preds, labels)
                print(f"Validation Accuracy: {accuracy:.4f}")

                # Save the model if accuracy improves
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(student.state_dict(), 'best_model.pth')
                    print(f"New best model saved with accuracy: {best_accuracy:.4f}")

                # add log of the test to writer
                writer.add_scalar('Accuracy/Test', accuracy)
                writer.add_scalar('Best Accuracy/Test', best_accuracy, (batch_idx) * (epoch + 1))


# def tes(epoch):
#     global best_acc
#     student.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     discriminator_loss = 0
#     crossentropy_loss = 0
#     contrastive_loss = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             num_batches = len(testloader)
#             total += targets.size(0)
#             inputs, targets = inputs.to(device), targets.to(device)
#
#             # Get output from student model
#             outputs = student(inputs)
#             # Get teacher model
#             teacher = teacher_selector(teachers)
#             # Get output from teacher model
#             answers = teacher(inputs)
#             # Select output from student and teacher
#             outputs, answers = output_selector(outputs, answers, eval(args.out_layer))
#             # Calculate loss between student and teacher
#             loss = criterion(outputs, answers)
#             # # Calculate loss for discriminators
#             # d_loss = discriminators_criterion(outputs, answers)
#             # Calculates the cross entropy loss for classification in the last layer
#             # ce_loss = nn.CrossEntropyLoss()(outputs[-1], targets)
#             # Calculate contrastive loss
#             lenght = len(outputs)
#             cntrstv_loss = sum(contrastive_criterion_test(outputs[i], answers[i]) for i in range(lenght))
#
#             test_loss += loss.item()
#             # discriminator_loss += d_loss.item()
#             # crossentropy_loss += ce_loss.item()
#             contrastive_loss += cntrstv_loss.item()
#             _, predicted = outputs[-1].max(1)
#             correct += predicted.eq(targets).sum().item()
#
#             # # Progress bar for discriminator loss + similarity loss
#             # progress_bar(batch_idx, len(testloader), 'Lr: %.4e | G_Loss: %.3f | D_Loss: %.3f | Acc: %.3f%% (%d/%d)'
#             #     % (scheduler.get_lr()[0], test_loss / (batch_idx + 1), discriminator_loss / (batch_idx + 1), 100. * correct / total, correct, total))
#
#             # Progress bar for contrastive loss + similarity loss
#             progress_bar(batch_idx, len(testloader), 'Lr: %.4e | Similarity_Loss: %.3f | Contrastive_Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                 % (scheduler.get_lr()[0], test_loss / (batch_idx + 1), contrastive_loss / (batch_idx + 1), 100. * correct / total, correct, total))
#
#             # Progress bar for contrastive loss
#             # progress_bar(batch_idx, len(testloader), 'Lr: %.4e | Contrastive_Loss: %.3f | Acc: %.3f%% (%d/%d)'
#             #     % (scheduler.get_lr()[0], contrastive_loss / (batch_idx + 1), 100. * correct / total, correct, total))
#
#         best_acc = max(100. * correct / total, best_acc)
#         # add log of the test to writer
#         writer.add_scalar('Accuracy/Test', 100. * correct / total, epoch)
#         writer.add_scalar('Best Accuracy/Test', best_acc, epoch)
#         writer.add_scalar('Similarity_loss/Test', test_loss / (batch_idx + 1), epoch)
#         # writer.add_scalar('Generator_loss/Test', test_loss / (batch_idx + 1), epoch)
#         writer.add_scalar('Contrastive_loss/Test', contrastive_loss / (batch_idx + 1), epoch)
#         # writer.add_scalar('Discriminator_loss/Test', discriminator_loss / (batch_idx + 1), epoch)
#
#     # Save checkpoint (the best accuracy).
#     if best_acc == (100. * correct / total):
#         print('Saving..')
#         state = {
#             'net': student.state_dict(),
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         # FILE_PATH = './checkpoint' + '/' + "_".join(args.teachers) + '-generator'
#         FILE_PATH = './checkpoint' + '/' + "_teachers_".join(args.teachers) + "_student_"+ args.student + '_loss_similarity+contrastive' + args.run
#         if os.path.isdir(FILE_PATH):
#             # print 'dir exists'generator
#             pass
#         else:
#             # print 'dir not exists'
#             os.mkdir(FILE_PATH)
#         # save_name = './checkpoint' + '/' + "_".join(args.teachers) + '-generator/ckpt.t7'
#         save_name = './checkpoint' + '/' + "_teachers_".join(args.teachers) + "_student_"+ args.student + '_loss_similarity+contrastive'+ args.run +'/ckpt.t7'
#         torch.save(state, save_name)


for epoch in range(start_epoch, start_epoch+args.epochs*(len(teachers))):
    train(epoch)
    # tes(epoch)
# Close the SummaryWriter when done
writer.close()

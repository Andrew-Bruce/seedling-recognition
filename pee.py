#!/usr/bin/env python
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
import csv
import torch
from torch import nn
import torchvision
import PIL


DEVICE = "cpu" #change later
INPUT_SHAPE = (28, 28)
NUM_CLASSES = 12
NUM_EPOCHS = 50
BATCH_SIZE = 100
LEARNING_RATE = 0.001
ANDY_AI_FILE = '陈功的AI_saved.ckpt'

IMAGE_SIZE = 224

andys_transpose= torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(IMAGE_SIZE),
    torchvision.transforms.ToTensor(),
])



class AndysDataSet(torch.utils.data.Dataset):
    def __init__(self, data_csv_filename, transform=None):
        self.csv_file = open(data_csv_filename, mode='rt', encoding="utf-8")
        self.csv_rows = list(csv.reader(self.csv_file, dialect="unix"))
        self.do_this_transform = transform
    def __len__(self):
        return len(self.csv_rows)
    def __getitem__(self, num):
        row = self.csv_rows[num]
        filename = row[1]
        image = PIL.Image.open(filename).convert('RGB')
        if self.do_this_transform is not None:
            image = self.do_this_transform(image)
        label = int(row[0])
        return image, label

def make_datasets(path):
    train_dataset = AndysDataSet(path, andys_transpose)
    return train_dataset

def make_loaders(train_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    return train_loader


class AndyNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.complicated_stuff = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), (1, 1), padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 1)),
            nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 32, (3, 3), (1, 1)),
            nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(93312, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, data):
        output_logits = self.complicated_stuff(data)
        return output_logits

def train(model, train_loader, criterion, optimizer):
    total_batches = len(train_loader)
    for epoch in range(NUM_EPOCHS):
        for batch_num, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            #back propigation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_num+1}/{total_batches}], Loss: {loss.item()}')

def main():
    print("making datasets")
    train_dataset = make_datasets('./data_labels.csv')
    print("making loaders")
    train_loader  = make_loaders(train_dataset)

    load_from_file = False

    if load_from_file:
        print(f"loading AI from file \"{ANDY_AI_FILE}\"")
        andy_ai = AndyNeuralNetwork()
        andy_ai.load_state_dict(torch.load(ANDY_AI_FILE))
        andy_ai.eval()
    else:
        print("creating new AI")
        andy_ai = AndyNeuralNetwork()
    print(f"sending AI to device {DEVICE}")
    andy_ai = andy_ai.to(DEVICE)
    print("creating loss function")
    criterion = nn.CrossEntropyLoss()#softmax
    print("creating optimizer")
    optimizer = torch.optim.SGD(andy_ai.parameters(), lr=LEARNING_RATE)
    print("Training AI")
    train(andy_ai, train_loader, criterion, optimizer)
    print(f"saving AI to file \"{ANDY_AI_FILE}\"")
    torch.save(andy_ai.state_dict(), ANDY_AI_FILE)


main()

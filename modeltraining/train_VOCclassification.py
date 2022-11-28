import torch
import torchvision
# tensorboard
from torch.utils.tensorboard import SummaryWriter
from UTILS.mydataset import VOCclassificationDataSet
from torchvision import transforms
from torch.utils.data import DataLoader

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# train dataset
train_dataset = VOCclassificationDataSet(voc_root="../dataset/VOCdevkit/VOC2012", transforms=data_transform,
                                         txt_name="train.txt")

# train dataloader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

# test dataset
test_dataset = VOCclassificationDataSet(voc_root="../dataset/VOCdevkit/VOC2012", transforms=data_transform,
                                        txt_name="val.txt")

# test dataloader
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ResNet50
model = torchvision.models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(2048, 21)
model.to(device)

# loss function
loss_func = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epoches = 10

# resume
# checkpoint = torch.load('./models/resnet50_voc_epoch_10.pth', map_location="cpu")
# model.load_state_dict(checkpoint["model"])
# optimizer.load_state_dict(checkpoint["optimizer"])
# epoch = checkpoint["epoch"]
# loss = checkpoint["loss"]
# acc=checkpoint["acc"]
# print("checkpoint acc = ",acc)


# tensorboard
writer = SummaryWriter(log_dir="./logs", comment="resnet50_voc")


# train
for epoch in range(epoches):

    model.train()
    loss_sum = 0
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # forward
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss_sum += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # Training progress bar
        print("\rEpoch: {}/{} | Step: {}/{} | Loss: {:.4f}".format(epoch + 1, epoches, i + 1, len(train_dataloader),
                                                                   loss.item()), end="")

    # tensorboard epoch loss
    writer.add_scalar('Train/Loss', loss_sum / len(train_dataloader), epoch)


    # loss average
    loss_avg = loss_sum / len(train_dataloader)
    print(" | Loss_avg: {:.4f}".format(loss_avg))

    # test
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:

            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # test progress bar
            print("\rTest: {}/{}".format(total, len(test_dataloader)), end="")

    print("Accuracy of the test images: {} %".format(100 * correct / total))

    # tensorboard epoch acc
    writer.add_scalar('Test/Acc', 100 * correct / total, epoch)


    # save checkpoint
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss_avg,
        "acc": 100 * correct / total
    }, "./models/resnet50_voc_epoch_{}.pth".format(epoch + 1))

writer.close()


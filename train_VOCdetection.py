import torch

from UTILS import presets
from UTILS.mydataset import VOCDetectionDataSet
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from UTILS.engine import evaluate, train_one_epoch
from torch.utils.tensorboard import SummaryWriter

# parameters
resume = None
batch_size = 1
lr = 0.002
epochs = 30

train_dataset = VOCDetectionDataSet(voc_root="../dataset/VOCdevkit/VOC2012/", txt_name="train.txt",
                                    transforms=presets.DetectionPresetTrain(data_augmentation='ssd'))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                              collate_fn=train_dataset.collate_fn)

val_dataset = VOCDetectionDataSet(voc_root="../dataset/VOCdevkit/VOC2012/", txt_name="val.txt",
                                  transforms=presets.DetectionPresetEval())

val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=val_dataset.collate_fn)

# ssd model

model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)


model.head.classification_head = SSDClassificationHead([512, 1024, 512, 256, 256, 256],
                                                       model.anchor_generator.num_anchors_per_location(), 21)

# optimizer

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1)

device = torch.device('cuda')

model.to(device)

start_epoch = 0

if resume is not None:
    print('In resume training.')
    checkpoint = torch.load(resume, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    start_epoch = checkpoint["epoch"] + 1

# tensorboard
writer = SummaryWriter(log_dir="./ssdlogs", comment="ssd_voc")

for epoch in range(start_epoch, epochs):
    metric_log, loss, classloss, boxloss = train_one_epoch(model, optimizer, train_dataloader, device, epoch, 10)
    lr_scheduler.step()
    coco_evaluator = evaluate(model, val_dataloader, device=device)

    # checkpoint
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "map": coco_evaluator.coco_eval["bbox"].stats[0]
    }, "./models/ssd300_vgg16_voc_epoch_{}.pth".format(epoch))

    # tensorboard
    writer.add_scalar("train/totalloss", loss, epoch)
    writer.add_scalar("train/classloss", classloss, epoch)
    writer.add_scalar("train/boxloss", boxloss, epoch)

    # VOC map@0.5
    writer.add_scalar("val/map", coco_evaluator.coco_eval["bbox"].stats[0], epoch)

    # VOC map@  0.5:0.95
    writer.add_scalar("val/map_0.5:0.95", coco_evaluator.coco_eval["bbox"].stats[1], epoch)



writer.close()

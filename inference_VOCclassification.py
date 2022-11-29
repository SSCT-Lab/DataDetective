import json

import torch
import torchvision.models
from torch.utils.data import DataLoader

from UTILS.mydataset import inference_VOCGt_classificationDataSet, inference_VOCinf_classificationDataSet
from torchvision import transforms


def inference_VOCclassification(dataloader, inference_type):
    model_path = './models/resnet50_voc_epoch_9.pth'
    # load model
    modelState = torch.load(model_path, map_location="cpu")
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Linear(2048, 21)
    model.load_state_dict(modelState["model"])
    model.eval()
    results = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images, targets = data

            outputs = model(images)
            # softmax outputs
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            # progress bar
            print("\rInference: {}/{}".format(i + 1, len(dataloader)), end="")

            # save softmax outputs image_name category_id boxes
            for j in range(len(predicted)):
                if inference_type == "gt":
                    content_dic = {
                        "image_name": targets["image_name"][j],
                        "full_scores": outputs[j].cpu().numpy().tolist(),
                        "detectiongt_category_id": int(targets["category_id"][j]),
                        "bbox": targets["boxes"][j].numpy().tolist(),
                    }
                    results.append(content_dic)
                elif inference_type == "inf":
                    content_dic = {
                        "image_name": targets["image_name"][j],
                        "full_scores": outputs[j].cpu().numpy().tolist(),
                        "detectioninf_category_id": int(targets["category_id"][j]),
                        "bbox": targets["boxes"][j].numpy().tolist(),
                    }
                    results.append(content_dic)
            print(results)
            break
    return results


if __name__ == '__main__':
    inference_type = 'inf'

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    vocgtdataset = inference_VOCGt_classificationDataSet(voc_root="./dataset/VOCdevkit/VOC2012",
                                                         transforms=data_transform,
                                                         txt_name="val.txt")
    vocgtdataloader = DataLoader(vocgtdataset, batch_size=1, shuffle=False, num_workers=0)

    vocinfdataset = inference_VOCinf_classificationDataSet(voc_root="./dataset/VOCdevkit/VOC2012",
                                                           inferences_root="./data/ssd_VOCval_inferences.json",
                                                           transforms=data_transform)
    vocinfdataloader = DataLoader(vocinfdataset, batch_size=1, shuffle=False, num_workers=0)

    if inference_type == 'gt':
        results = inference_VOCclassification(vocgtdataloader, inference_type)
        json_str = json.dumps(results, indent=4)
        with open('./data/classification_VOCgt_inferences.json', 'w') as json_file:
            json_file.write(json_str)

    elif inference_type == 'inf':
        results = inference_VOCclassification(vocinfdataloader, inference_type)
        json_str = json.dumps(results, indent=4)
        with open('./data/classification_VOCinf_inferences.json', 'w') as json_file:
            json_file.write(json_str)

    else:
        print("inference_type error")

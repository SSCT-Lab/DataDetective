import json

import torch
import torchvision.models
from torch.utils.data import DataLoader

from UTILS.mydataset import inference_VOCGt_classificationDataSet, inference_VOCinf_classificationDataSet, \
    inference_VOCgtfault_classificationDataSet
from torchvision import transforms

from UTILS.parameters import parameters


def inference_VOCclassification(dataloader, inference_type):
    model_path = './models/resnet50_voc_epoch_9.pth'
    # load model
    modelState = torch.load(model_path, map_location="cpu")
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Linear(2048, 21)
    model.load_state_dict(modelState["model"])
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    results = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images, targets = data

            outputs = model(images.to(device))
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

                # inference_type == 'class fault' or 'location fault' or 'redundancy fault' or 'missing fault'
                else:
                    content_dic = {
                        "image_name": targets["image_name"][j],
                        "full_scores": outputs[j].cpu().numpy().tolist(),
                        "detectiongt_category_id": int(targets["category_id"][j]),
                        "bbox": targets["boxes"][j].numpy().tolist(),
                        "fault_type": targets["fault_type"][j].item(),
                    }
                    results.append(content_dic)

    return results


params = parameters()
if __name__ == '__main__':
    inference_type = 'mixed fault'
    modeltype = 'frcnn'

    detection_results = {
        "ssd": "./data/detection_results/ssd_VOCval_inferences.json",
        "frcnn": "./data/detection_results/frcnn_VOCval_inferences.json",
    }
    clssification_results = {
        "ssd": './data/classification_results/classification_VOCssdinf' + str(params.m_t) + '_inferences.json',
        "frcnn": './data/classification_results/classification_VOCfrcnninf' + str(params.m_t) + '_inferences.json',
    }

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataloader = None

    if inference_type == 'gt':
        dataset = inference_VOCGt_classificationDataSet(voc_root="./dataset/VOCdevkit/VOC2012",
                                                        transforms=data_transform,
                                                        txt_name="val.txt")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    elif inference_type == 'inf':
        dataset = inference_VOCinf_classificationDataSet(voc_root="./dataset/VOCdevkit/VOC2012",
                                                         inferences_root=detection_results[modeltype],
                                                         transforms=data_transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    elif inference_type == 'class fault':
        dataset = inference_VOCgtfault_classificationDataSet(voc_root="./dataset/VOCdevkit/VOC2012",
                                                             fault_type="class fault",
                                                             transforms=data_transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    elif inference_type == 'location fault':
        dataset = inference_VOCgtfault_classificationDataSet(voc_root="./dataset/VOCdevkit/VOC2012",
                                                             fault_type="location fault",
                                                             transforms=data_transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    elif inference_type == 'redundancy fault':
        dataset = inference_VOCgtfault_classificationDataSet(voc_root="./dataset/VOCdevkit/VOC2012",
                                                             fault_type="redundancy fault",
                                                             transforms=data_transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    elif inference_type == 'missing fault':
        dataset = inference_VOCgtfault_classificationDataSet(voc_root="./dataset/VOCdevkit/VOC2012",
                                                             fault_type="missing fault",
                                                             transforms=data_transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    elif inference_type == 'mixed fault':
        dataset = inference_VOCgtfault_classificationDataSet(voc_root="./dataset/VOCdevkit/VOC2012",
                                                             fault_type="mixed fault",
                                                             transforms=data_transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    if inference_type == 'gt':
        results = inference_VOCclassification(dataloader, inference_type)
        json_str = json.dumps(results, indent=4)
        with open('./data/classification_results/classification_VOCgt_inferences.json', 'w') as json_file:
            json_file.write(json_str)

    elif inference_type == 'inf':
        results = inference_VOCclassification(dataloader, inference_type)
        json_str = json.dumps(results, indent=4)
        with open(clssification_results[modeltype], 'w') as json_file:
            json_file.write(json_str)

    elif inference_type == 'class fault':
        results = inference_VOCclassification(dataloader, inference_type)
        json_str = json.dumps(results, indent=4)
        with open('./data/classification_results/classification_VOCgtclassfault_inferences.json', 'w') as json_file:
            json_file.write(json_str)
    elif inference_type == 'location fault':
        results = inference_VOCclassification(dataloader, inference_type)
        json_str = json.dumps(results, indent=4)
        with open('./data/classification_results/classification_VOCgtlocationfault_inferences.json', 'w') as json_file:
            json_file.write(json_str)
    elif inference_type == 'redundancy fault':
        results = inference_VOCclassification(dataloader, inference_type)
        json_str = json.dumps(results, indent=4)
        with open('./data/classification_results/classification_VOCgtredundancyfault_inferences.json',
                  'w') as json_file:
            json_file.write(json_str)

    elif inference_type == 'missing fault':
        results = inference_VOCclassification(dataloader, inference_type)
        json_str = json.dumps(results, indent=4)
        with open('./data/classification_results/classification_VOCgtmissingfault_inferences.json', 'w') as json_file:
            json_file.write(json_str)
    elif inference_type == 'mixed fault':
        results = inference_VOCclassification(dataloader, inference_type)
        json_str = json.dumps(results, indent=4)
        with open('./data/classification_results/classification_VOCgtmixedfault_inferences.json', 'w') as json_file:
            json_file.write(json_str)

    else:
        print("inference_type error")

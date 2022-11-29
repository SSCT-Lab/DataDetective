import json

import torch
from tqdm import tqdm

from modelsCodes.models.detection import ssd300_vgg16
from UTILS.mydataset import inferenceVOCDetectionDataSet
from UTILS import presets
from UTILS.engine import evaluate
from UTILS.cal_voc_map import cal_voc_map

model_path = 'models/ssd300_vgg16_voc_epoch_29.pth'

# load model
modelState = torch.load(model_path, map_location="cpu")
model = ssd300_vgg16(num_classes=21)
model.load_state_dict(modelState["model"])

val_dataset = inferenceVOCDetectionDataSet(voc_root="./dataset/VOCdevkit/VOC2012",
                                           transforms=presets.DetectionPresetEval(),
                                           txt_name="val.txt")

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,
                                             collate_fn=val_dataset.collate_fn)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# inference

results = []
model.eval()
with torch.no_grad():
    for image, targets in tqdm(val_dataloader):

        image = list(img.to(device) for img in image)



        outputs = model(image)
        # instances results
        for i, prediction in enumerate(outputs):
            cat_ids = prediction["labels"].cpu()
            bboxs = prediction["boxes"].cpu().numpy().tolist()
            scores = prediction['scores'].cpu()
            full_score = prediction['full_scores'].cpu().numpy().tolist()
            for j in range(prediction["labels"].shape[0]):
                content_dic = {
                    "image_name": targets[i]["image_name"],
                    "image_id": int(targets[i]["image_id"].numpy()[0]),
                    "category_id": int(cat_ids[j]),
                    "bbox": bboxs[j],
                    "score": float(scores[j]),
                    "full_scores": full_score[j],
                }
                results.append(content_dic)

    json_str = json.dumps(results, indent=4)
    with open('data/detection_results/ssd_VOCval_inferences.json', 'w') as json_file:
        json_file.write(json_str)

# load results
with open('data/detection_results/ssd_VOCval_inferences.json', 'r') as f:
    results = json.load(f)

# check max(full_score) == score
# for i in range(len(results)):
#     print(results[i]["score"],max(results[i]["full_scores"][1:]))
    # assert results[i]["score"] == max(results[i]["full_scores"][1:]), "score != max(full_score)"

cal_voc_map(model, results, val_dataset)

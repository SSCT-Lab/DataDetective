import json

import fiftyone as fo
import fiftyone.brain as fob
import torch
from fiftyone import ViewField as F
from torchvision.ops import boxes as box_ops
# transfrom x1,y1,x2,y2 to x1,y1,w,h relative values
from UTILS.parameters import parameters

params = parameters()
fault_type_dict = parameters().fault_type

# convert fault_type_dict to number2fault
number2fault = {}
for key in fault_type_dict.keys():
    number2fault[fault_type_dict[key]] = key


class Fiftyone:
    def __init__(self, gt_path='./data/fault_annotations/VOCval_mixedfault.json'
                 , dec_path='./data/detection_results/frcnn_VOCval_inferences.json'
                 , image_path='./dataset/VOCdevkit/VOC2012/JPEGImages/'):
        self.gt_path = gt_path
        self.dec_path = dec_path
        self.image_path = image_path

        fault_num = {
            'no fault': 0,
            'class fault': 0,
            'location fault': 0,
            'redundancy fault': 0,
            'missing fault': 0,
        }
        with open(self.gt_path, 'r') as f:
            gt = json.load(f)
        for i in gt:
            fault_num[number2fault[i["fault_type"]]] += 1

        self.fault_num = fault_num

        missing_list = []
        for i in range(len(gt)):
            if gt[i]["fault_type"] == fault_type_dict['missing fault']:
                missing_list.append(gt[i])
        # transform missing list to {imagename:[]} format dict
        missing_dict = {}
        for i in range(len(missing_list)):
            if missing_list[i]["image_name"] in missing_dict:
                missing_dict[missing_list[i]["image_name"]].append(missing_list[i])
            else:
                missing_dict[missing_list[i]["image_name"]] = [missing_list[i]]

        self.missing_dict = missing_dict

    def run(self):
        with open(self.gt_path, 'r') as f:
            full_list = json.load(f)

        no_missing_list = []
        for i in range(len(full_list)):
            if full_list[i]["fault_type"] != fault_type_dict['missing fault']:
                no_missing_list.append(full_list[i])
        # transform full_list list to {imagename:[]} format dict
        image_dict = {}
        for item in no_missing_list:
            if item['image_name'] not in image_dict.keys():
                image_dict[item['image_name']] = []
            image_dict[item['image_name']].append(item)

        samples = []
        for imagename in image_dict.keys():
            sample = fo.Sample(filepath=self.image_path + imagename)

            # add ground truth labels
            detections = []
            for item in image_dict[imagename]:
                label = item['labels']
                bounding_box = self.xyxy2xywh_relative(item['boxes'], item['image_size'])
                fault_type = item['fault_type']
                detections.append(
                    fo.Detection(label=str(label), bounding_box=bounding_box, fault_type=fault_type,
                                 imagename=imagename, image_size=item['image_size'])
                )
            sample["ground_truth"] = fo.Detections(detections=detections)

            samples.append(sample)

        dataset = fo.Dataset("my-detection-dataset")
        dataset.add_samples(samples)

        with open(self.dec_path, 'r') as f:
            dec = json.load(f)
        # transform dec list to {imagename:[]} format dict
        dec_image_dict = {}

        for item in dec:
            if item['image_name'] not in dec_image_dict.keys():
                dec_image_dict[item['image_name']] = []
            dec_image_dict[item['image_name']].append(item)

        for sample in dataset:
            filepath = sample.filepath
            image_name = filepath.split('\\')[-1]

            detections = []
            if image_name in dec_image_dict.keys():
                for item in dec_image_dict[image_name]:
                    label = item['category_id']
                    bounding_box = self.xyxy2xywh_relative(item['bbox'], image_dict[image_name][0]['image_size'])
                    confidence = item["score"]
                    detections.append(
                        fo.Detection(label=str(label), bounding_box=bounding_box, confidence=confidence,
                                     imagename=image_name, image_size=image_dict[image_name][0]['image_size'])
                    )
                sample["predictions"] = fo.Detections(detections=detections)

                sample.save()
        session = fo.launch_app(dataset, desktop=True)

        fob.compute_mistakenness(dataset, "predictions", label_field="ground_truth")

        self.RQ1(dataset)

    def RQ1(self, dataset):
        print("RQ1")
        possible_spurious_list = dataset.filter_labels("ground_truth", F("possible_spurious") == True)

        ps_list_length = len(possible_spurious_list)

        ps_class_fault_num = 0
        ps_location_fault_num = 0
        ps_redundancy_fault_num = 0

        for sample in possible_spurious_list:
            for detection in sample.ground_truth.detections:
                if detection.possible_spurious:
                    if detection["fault_type"] == fault_type_dict['class fault']:
                        ps_class_fault_num += 1
                    elif detection["fault_type"] == fault_type_dict['location fault']:
                        ps_location_fault_num += 1
                    elif detection["fault_type"] == fault_type_dict['redundancy fault']:
                        ps_redundancy_fault_num += 1

        totalfault_num = self.fault_num['class fault'] + self.fault_num['location fault'] + self.fault_num[
            'redundancy fault'] + self.fault_num['missing fault']

        print('\n=========possible_spurious==============')

        print('FaultDetectionRate - ' + 'class fault' + ': ', ps_class_fault_num / ps_list_length)

        print('Inclusiveness - ' + 'class fault' + ': ', ps_class_fault_num / self.fault_num['class fault'])

        # print Dividing Line
        print('----------------------------------------')

        print('FaultDetectionRate - ' + 'location fault' + ': ', ps_location_fault_num / ps_list_length)

        print('Inclusiveness - ' + 'location fault' + ': ', ps_location_fault_num / self.fault_num['location fault'])

        print('----------------------------------------')

        print('FaultDetectionRate - ' + 'redundancy fault' + ': ', ps_redundancy_fault_num / ps_list_length)

        print('Inclusiveness - ' + 'redundancy fault' + ': ',
              ps_redundancy_fault_num / self.fault_num['redundancy fault'])

        print('----------------------------------------')

        print('FaultDetectionRate - all fault: ',
              (ps_class_fault_num + ps_location_fault_num + ps_redundancy_fault_num) / ps_list_length)
        print('Inclusiveness - all fault: ',
              (ps_class_fault_num + ps_location_fault_num + ps_redundancy_fault_num) / totalfault_num)

        print('========================================')

        possible_missing_list = dataset.filter_labels("predictions", F("possible_missing") == True)

        pm_list_length = len(possible_missing_list)

        pm_missing_fault_num = 0

        for sample in possible_missing_list:
            for detection in sample.predictions.detections:
                if detection.possible_missing:
                    image_name = detection["imagename"]
                    image_size = detection["image_size"]
                    if image_name in self.missing_dict.keys():
                        box = self.xywh_relative2xyxy(detection["bounding_box"], image_size)

                        missing_boxes = [j["boxes"] for j in self.missing_dict[image_name]]
                        missing_max_IoU = torch.max(self.cal_IoU(box, missing_boxes))
                        if missing_max_IoU > 0.5:
                            pm_missing_fault_num += 1

        print('\n=========possible_missing==============')

        print('FaultDetectionRate - ' + 'missing fault' + ': ', pm_missing_fault_num / pm_list_length)

        print('Inclusiveness - ' + 'missing fault' + ': ', pm_missing_fault_num / self.fault_num['missing fault'])

        print('FaultDetectionRate - all fault: ', pm_missing_fault_num / pm_list_length)

        print('Inclusiveness - all fault: ', pm_missing_fault_num / totalfault_num)

        print('================End=====================')

    def cal_IoU(self, X, Y):
        return box_ops.box_iou(torch.tensor([X]), torch.tensor(Y))

    def xyxy2xywh_relative(self, box, image_size):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        return [x1 / image_size[0], y1 / image_size[1], w / image_size[0], h / image_size[1]]

    def xywh_relative2xyxy(self, box, image_size):

        x1 = box[0] * image_size[0]
        y1 = box[1] * image_size[1]
        w = box[2] * image_size[0]
        h = box[3] * image_size[1]
        return [x1, y1, x1 + w, y1 + h]


if __name__ == '__main__':
    FO = Fiftyone()
    FO.run()

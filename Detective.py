import json
import os

import torch
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from tqdm import tqdm

from UTILS.parameters import parameters
from torchvision.ops import boxes as box_ops

params = parameters()
fault_type_dict = parameters().fault_type

# convert fault_type_dict to number2fault
number2fault = {}
for key in fault_type_dict.keys():
    number2fault[fault_type_dict[key]] = key


class FaultDetective:
    def __init__(self, clsgt_path='./data/classification_results/classification_VOCgtmixedfault_inferences.json'
                 , det_path='./data/detection_results/frcnn_VOCval_inferences.json'
                 , clsinf_path='./data/classification_results/classification_VOCfrcnninf' + str(
                params.m_t) + '_inferences.json'
                 , gt_path='./data/fault_annotations/VOCval_mixedfault.json'):

        self.clsgt_path = clsgt_path
        self.det_path = det_path
        self.clsinf_path = clsinf_path
        self.gt_path = gt_path

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

    def Wrongcls(self):
        print("Wrong class")
        cls_gt = []  # ./data/classification/classification_VOCgt_inferences.json
        dec = []  # ./data/detection_results/ssd_VOCval_inferences.json
        with open(self.clsgt_path, 'r') as f:
            cls_gt = json.load(f)
        with open(self.det_path, 'r') as f:
            dec = json.load(f)
        # get inference results score > m_t

        dec = [i for i in dec if i["score"] > params.m_t]

        # print cls_gt length and dec length
        print("cls_gt length: ", len(cls_gt))
        print("dec length: ", len(dec))

        # transform dec to {imagename:[]} format dict
        dec_dict = {}
        for i in range(len(dec)):
            if dec[i]["image_name"] in dec_dict:
                dec_dict[dec[i]["image_name"]].append(dec[i])
            else:
                dec_dict[dec[i]["image_name"]] = [dec[i]]

        results = []
        for i in tqdm(range(len(cls_gt))):
            # image name of this instance
            image_name = cls_gt[i]["image_name"]
            box = cls_gt[i]["bbox"]
            cls_full_score = cls_gt[i]["full_scores"]
            detectiongt_category = cls_gt[i]["detectiongt_category_id"]
            fault_type = cls_gt[i]["fault_type"]
            # if this image is in dec_dict
            if image_name in dec_dict:
                # boxes of this image in dec_dict
                boxes = [j["bbox"] for j in dec_dict[image_name]]
                # IoU of this instance and boxes in dec_dict
                IoU = self.cal_IoU(box, boxes)
                # if IoU > params.t_f
                if torch.max(IoU) > params.t_f:
                    # cal cross_entropy loss of cls_full_score and detectiongt_category
                    loss = self.cross_entropy_loss(cls_full_score, detectiongt_category)

                    # save this instance and loss to results
                    results.append(
                        {"image_name": image_name, "wrongcls_loss": loss, "bbox": box, "full_scores": cls_full_score,
                         "detectiongt_category_id": detectiongt_category, "fault_type": fault_type})

        # sort results by loss from large to small
        results = sorted(results, key=lambda x: x["wrongcls_loss"], reverse=True)

        self.plt_cruve(results, 'class fault')

    def PoorLoc(self):
        cls_gt = []  # ./data/classification/classification_VOCgt_inferences.json
        cls_inf = []  # ./data/classification/classification_VOCssdinf0.5_inferences.json
        dec = []  # ./data/detection_results/ssd_VOCval_inferences.json
        with open(self.clsgt_path, 'r') as f:
            cls_gt = json.load(f)
        with open(self.clsinf_path, 'r') as f:
            cls_inf = json.load(f)
        with open(self.det_path, 'r') as f:
            dec = json.load(f)
        # get inference results score > m_t

        dec = [i for i in dec if i["score"] > params.m_t]

        # print cls_inf length, cls_gt length and dec length
        print("cls_inf length: ", len(cls_inf))
        print("cls_gt length: ", len(cls_gt))
        print("dec length: ", len(dec))

        # transform dec to {imagename:[]} format dict
        dec_dict = {}
        for i in range(len(dec)):
            if dec[i]["image_name"] in dec_dict:
                dec_dict[dec[i]["image_name"]].append(dec[i])
            else:
                dec_dict[dec[i]["image_name"]] = [dec[i]]

        # transform cls_inf to {imagename:[]} format dict
        cls_inf_dict = {}
        for i in range(len(cls_inf)):
            if cls_inf[i]["image_name"] in cls_inf_dict:
                cls_inf_dict[cls_inf[i]["image_name"]].append(cls_inf[i])
            else:
                cls_inf_dict[cls_inf[i]["image_name"]] = [cls_inf[i]]

        results = []

        # for every instance in cls_gt
        for i in tqdm(range(len(cls_gt))):
            # image name of this instance
            image_name = cls_gt[i]["image_name"]
            box = cls_gt[i]["bbox"]
            cls_gt_full_score = cls_gt[i]["full_scores"]
            detectiongt_category = cls_gt[i]["detectiongt_category_id"]
            fault_type = cls_gt[i]["fault_type"]

            # if this image is in dec_dict
            if image_name in dec_dict:
                assert len(dec_dict[image_name]) == len(
                    cls_inf_dict[image_name]), "dec_dict[image_name] != cls_inf_dict[image_name]"
                for j in range(len(dec_dict[image_name])):
                    assert dec_dict[image_name][j]["bbox"] == cls_inf_dict[image_name][j]["bbox"], "bbox not match"
                # boxes of this image in dec_dict
                boxes = [j["bbox"] for j in dec_dict[image_name]]
                categorys = [j["category_id"] for j in dec_dict[image_name]]
                # IoU of this instance and boxes in dec_dict
                IoU = self.cal_IoU(box, boxes)
                # max_IoU corresponding score

                max_IoU = torch.max(IoU)
                P_k = dec_dict[image_name][torch.argmax(IoU).item()]["score"]

                # max_IoU corresponding cls_inf
                cls_inf_item = cls_inf_dict[image_name][torch.argmax(IoU).item()]
                cls_max_p = torch.max(torch.tensor(cls_inf_item["full_scores"]))

                assert categorys[torch.argmax(IoU).item()] == cls_inf_item[
                    "detectioninf_category_id"], "category not match"

                # if t_b < max_IoU <= t_f and max_IoU_score > t_p
                if params.t_b < max_IoU <= params.t_f and P_k > params.t_p:
                    loss = max_IoU * cls_max_p
                    results.append({"image_name": image_name, "gtbox": box, "infbox": boxes[torch.argmax(IoU).item()],
                                    "gt_category": detectiongt_category,
                                    "cls_gt_category": torch.argmax(torch.tensor(cls_gt_full_score)).item(),
                                    "inf_category": categorys[torch.argmax(IoU).item()],
                                    "cls_inf_category": torch.argmax(torch.tensor(cls_inf_item["full_scores"])).item(),
                                    "poorloc_loss": loss,
                                    "fault_type": fault_type})
        # sort results by loss from large to small
        results = sorted(results, key=lambda x: x["poorloc_loss"], reverse=True)

        self.plt_cruve(results, 'location fault')

    def Redundancy(self):
        print("Redundancy")
        cls_gt = []  # ./data/classification/classification_VOCgt_inferences.json
        dec = []  # ./data/detection_results/ssd_VOCval_inferences.json
        with open(self.clsgt_path, 'r') as f:
            cls_gt = json.load(f)
        with open(self.det_path, 'r') as f:
            dec = json.load(f)
        # get inference results score > m_t
        params = parameters()
        dec = [i for i in dec if i["score"] > params.m_t]

        # print cls_gt length and dec length
        print("cls_gt length: ", len(cls_gt))
        print("dec length: ", len(dec))

        # transform dec to {imagename:[]} format dict
        dec_dict = {}
        for i in range(len(dec)):
            if dec[i]["image_name"] in dec_dict:
                dec_dict[dec[i]["image_name"]].append(dec[i])
            else:
                dec_dict[dec[i]["image_name"]] = [dec[i]]

        results = []
        # for every instance in cls_gt
        for i in tqdm(range(len(cls_gt))):
            # image name of this instance
            image_name = cls_gt[i]["image_name"]
            box = cls_gt[i]["bbox"]
            cls_full_score = cls_gt[i]["full_scores"]
            detectiongt_category = cls_gt[i]["detectiongt_category_id"]
            fault_type = cls_gt[i]["fault_type"]
            # if this image is in dec_dict
            if image_name in dec_dict:
                # boxes of this image in dec_dict
                boxes = [j["bbox"] for j in dec_dict[image_name]]
                # IoU of this instance and boxes in dec_dict
                IoU = self.cal_IoU(box, boxes)
                # if IoU < params.t_b
                if torch.max(IoU) < params.t_b:
                    # cal cross_entropy loss of cls_full_score and detectiongt_category
                    loss = self.cross_entropy_loss(cls_full_score, detectiongt_category)

                    # save this instance and loss to results
                    results.append(
                        {"image_name": image_name, "redundancy_loss": loss, "bbox": box, "full_scores": cls_full_score,
                         "detectiongt_category_id": detectiongt_category, "fault_type": fault_type})

        # sort results by loss from large to small
        results = sorted(results, key=lambda x: x["redundancy_loss"], reverse=True)

        self.plt_cruve(results, "redundancy fault")

    def Missing(self):
        print("Missing")
        cls_inf = []  # ./data/classification/classification_VOCssdinf0.5_inferences.json
        cls_gt = []  # ./data/lassification/classification_VOCgt_inferences.json
        with open(self.clsinf_path, 'r') as f:
            cls_inf = json.load(f)
        with open(self.clsgt_path, 'r') as f:
            cls_gt = json.load(f)

        # missing list
        missing_list = []
        with open(self.gt_path, 'r') as f:
            full_list = json.load(f)
        for i in range(len(full_list)):
            if full_list[i]["fault_type"] == fault_type_dict['missing fault']:
                missing_list.append(full_list[i])
        # transform missing list to {imagename:[]} format dict
        missing_dict = {}
        for i in range(len(missing_list)):
            if missing_list[i]["image_name"] in missing_dict:
                missing_dict[missing_list[i]["image_name"]].append(missing_list[i])
            else:
                missing_dict[missing_list[i]["image_name"]] = [missing_list[i]]

        # print cls_inf length and cls_gt length
        print("cls_inf length: ", len(cls_inf))
        print("cls_gt length: ", len(cls_gt))

        # transform cls_gt to {imagename:[]} format dict
        cls_gt_dict = {}
        for i in range(len(cls_gt)):
            if cls_gt[i]["image_name"] in cls_gt_dict:
                cls_gt_dict[cls_gt[i]["image_name"]].append(cls_gt[i])
            else:
                cls_gt_dict[cls_gt[i]["image_name"]] = [cls_gt[i]]

        results = []
        # for every instance in cls_inf
        for i in tqdm(range(len(cls_inf))):
            # image name of this instance
            image_name = cls_inf[i]["image_name"]
            box = cls_inf[i]["bbox"]
            cls_full_score = cls_inf[i]["full_scores"]
            detectioninf_category = cls_inf[i]["detectioninf_category_id"]
            fault_type = fault_type_dict['no fault']
            # if this image is in cls_gt_dict
            if image_name in cls_gt_dict:
                # boxes of this image in cls_gt_dict
                boxes = [j["bbox"] for j in cls_gt_dict[image_name]]
                # IoU of this instance and boxes in cls_gt_dict
                IoU = self.cal_IoU(box, boxes)
                # if IoU < params.t_f

                if torch.max(IoU) < params.t_f:
                    # loss = max(cls_full_score)
                    loss = torch.max(torch.tensor(cls_full_score))

                    # if this image is in missing_dict
                    if image_name in missing_dict:
                        missing_boxes = [j["boxes"] for j in missing_dict[image_name]]
                        # max IoU of this box and missing boxes
                        missing_max_IoU = torch.max(self.cal_IoU(box, missing_boxes))
                        if missing_max_IoU > 0.5:
                            fault_type = fault_type_dict['missing fault']

                    # save this instance and loss to results
                    results.append(
                        {"image_name": image_name, "missing_loss": loss, "bbox": box, "full_scores": cls_full_score,
                         "detectioninf_category_id": detectioninf_category, "fault_type": fault_type})

        # sort results by loss from large to small
        results = sorted(results, key=lambda x: x["missing_loss"], reverse=True)
        self.plt_cruve(results, "missing fault")

    def cal_IoU(self, X, Y):
        return box_ops.box_iou(torch.tensor([X]), torch.tensor(Y))

    def cross_entropy_loss(self, cls_full_score, detectiongt_category):
        loss = torch.nn.CrossEntropyLoss()(torch.tensor(cls_full_score), torch.tensor(detectiongt_category))
        # loss to float
        loss = loss.item()
        return loss

    def plt_cruve(self, results, fault_type):
        x = [i for i in range(len(results))]
        # y = accumulate the number of fault instances
        y = [0 for i in range(len(results))]

        # z = accumulate the number of any fault instances
        z = [0 for i in range(len(results))]

        fault_type_dict = params.fault_type
        for i in range(len(results)):
            if results[i]['fault_type'] == fault_type_dict[fault_type]:
                y[i] = y[i - 1] + 1

            else:
                y[i] = y[i - 1]

            if results[i]['fault_type'] != fault_type_dict['no fault']:
                z[i] = z[i - 1] + 1

            else:
                z[i] = z[i - 1]

        totalfault_num = self.fault_num['class fault'] + self.fault_num['location fault'] + self.fault_num[
            'redundancy fault'] + self.fault_num['missing fault']

        print('ratio:', y[-1] / self.fault_num[fault_type])
        print('ratio:', z[-1] / totalfault_num)

        plt.title(fault_type)
        plt.plot(x, y, label=fault_type)
        plt.plot(x, z, label='any fault')
        plt.xlabel('instances')
        plt.ylabel('fault instances')
        # draw the ratio with 3 floats in the center of the curve line
        plt.text(len(results) / 2, y[-1] / 2, fault_type + ' ratio: %.3f' % (y[-1] / self.fault_num[fault_type]),
                 fontsize=10)

        plt.text(len(results) / 2, z[-1] / 2 + 10, 'any fault ratio: %.3f' % (z[-1] / totalfault_num),
                 fontsize=10)
        print('FaultDetectionRate - ' + fault_type + ': ', y[-1] / len(results))
        print('Inclusiveness - ' + fault_type + ': ', y[-1] / self.fault_num[fault_type])

        print('FaultDetectionRate - any fault: ', z[-1] / len(results))
        print('Inclusiveness - any fault: ', z[-1] / totalfault_num)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    Fd = FaultDetective()
    Fd.Missing()

import json
import os

import torch
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from tqdm import tqdm

from UTILS.parameters import parameters
from torchvision.ops import boxes as box_ops

params = parameters()


def cal_IoU(X, Y):
    return box_ops.box_iou(torch.tensor([X]), torch.tensor(Y))


def cross_entropy_loss(cls_full_score, detectiongt_category):
    loss = torch.nn.CrossEntropyLoss()(torch.tensor(cls_full_score), torch.tensor(detectiongt_category))
    # loss to float
    loss = loss.item()
    return loss


def plt_cruve(results, fault_type, total_fault_num):
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

    print('ratio:', y[-1] / total_fault_num)
    print('ratio:', z[-1] / (total_fault_num * 4))

    plt.title(fault_type)
    plt.plot(x, y, label=fault_type)
    plt.plot(x, z, label='any fault')
    plt.xlabel('instances')
    plt.ylabel('fault instances')
    # draw the ratio with 3 floats in the center of the curve line
    plt.text(len(results) / 2, y[-1]/2, fault_type + ' ratio: %.3f' % (y[-1] / total_fault_num),
             fontsize=10)

    plt.text(len(results) / 2, z[-1]/2+10, 'any fault ratio: %.3f' % (z[-1] / (total_fault_num * 4)),
             fontsize=10)

    plt.legend()
    plt.show()


def Wrongcls():
    print("Wrong class")
    cls_gt = []  # ./data/classification/classification_VOCgt_inferences.json
    dec = []  # ./data/detection_results/ssd_VOCval_inferences.json
    with open('./data/classification_results/classification_VOCgtmixedfault_inferences.json', 'r') as f:
        cls_gt = json.load(f)
    with open('./data/detection_results/ssd_VOCval_inferences.json', 'r') as f:
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
    total_fault_num = 0
    fault_type_dict = params.fault_type
    for i in tqdm(range(len(cls_gt))):
        # image name of this instance
        image_name = cls_gt[i]["image_name"]
        box = cls_gt[i]["bbox"]
        cls_full_score = cls_gt[i]["full_scores"]
        detectiongt_category = cls_gt[i]["detectiongt_category_id"]
        fault_type = cls_gt[i]["fault_type"]
        if fault_type == fault_type_dict['class fault']:
            total_fault_num += 1
        # if this image is in dec_dict
        if image_name in dec_dict:
            # boxes of this image in dec_dict
            boxes = [j["bbox"] for j in dec_dict[image_name]]
            # IoU of this instance and boxes in dec_dict
            IoU = cal_IoU(box, boxes)
            # if IoU > params.t_f
            if torch.max(IoU) > params.t_f:
                # cal cross_entropy loss of cls_full_score and detectiongt_category
                loss = cross_entropy_loss(cls_full_score, detectiongt_category)

                # save this instance and loss to results
                results.append(
                    {"image_name": image_name, "wrongcls_loss": loss, "bbox": box, "full_scores": cls_full_score,
                     "detectiongt_category_id": detectiongt_category, "fault_type": fault_type})

    # sort results by loss from large to small
    results = sorted(results, key=lambda x: x["wrongcls_loss"], reverse=True)

    plt_cruve(results, 'class fault', total_fault_num)

    # # read class_indict
    # class_dict = {}
    # json_file = os.path.join('dataset/VOCdevkit/VOC2012', 'pascal_voc_classes.json')
    # assert os.path.exists(json_file), 'json file not found'
    # with open(json_file, 'r') as fp:
    #     class_dict = json.load(fp)
    # # convert class_indict to {class_id:class_name}
    # class_dict = dict((val, key) for key, val in class_dict.items())

    # save imgages of top 100 results with box and class_name1 and class_name2
    # for i in range(100):
    #     image_name = results[i]["image_name"]
    #     box = results[i]["bbox"]
    #     full_scores = results[i]["full_scores"]
    #     detectiongt_category_id = results[i]["detectiongt_category_id"]
    #     loss = results[i]["wrongcls_loss"]
    #
    #     image = Image.open("dataset/VOCdevkit/VOC2012/JPEGImages/" + image_name).convert("RGB")
    #
    #     class_name1 = class_dict[detectiongt_category_id]
    #     class_name2 = class_dict[torch.argmax(torch.tensor(full_scores)).item()]
    #
    #     # draw image with bounding box and labels
    #     plt.gca().add_patch(
    #         plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red',
    #                       linewidth=2))
    #     plt.gca().text(box[0], box[1] - 2, '{:s} guess: {:s}{:.3f}'.format(class_name1, class_name2, loss),
    #                    bbox=dict(facecolor='blue', alpha=0.5), fontsize=12, color='white')
    #     # 关闭坐标轴 去除图片空白部分
    #     plt.axis('off')
    #     plt.imshow(image)
    #     plt.savefig("data/detective_results/wrongcls/" + str(i) + '_' + image_name, bbox_inches='tight', pad_inches=0)
    #     plt.cla()
    #
    # print("results length: ", len(results))


def PoorLoc():
    cls_gt = []  # ./data/classification/classification_VOCgt_inferences.json
    cls_inf = []  # ./data/classification/classification_VOCinf0.5_inferences.json
    dec = []  # ./data/detection_results/ssd_VOCval_inferences.json
    with open('./data/classification_results/classification_VOCgtmixedfault_inferences.json', 'r') as f:
        cls_gt = json.load(f)
    with open('./data/classification_results/classification_VOCinf' + str(params.m_t) + '_inferences.json', 'r') as f:
        cls_inf = json.load(f)
    with open('./data/detection_results/ssd_VOCval_inferences.json', 'r') as f:
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
    total_fault_num = 0
    fault_type_dict = params.fault_type
    # for every instance in cls_gt
    for i in tqdm(range(len(cls_gt))):
        # image name of this instance
        image_name = cls_gt[i]["image_name"]
        box = cls_gt[i]["bbox"]
        cls_gt_full_score = cls_gt[i]["full_scores"]
        detectiongt_category = cls_gt[i]["detectiongt_category_id"]
        fault_type = cls_gt[i]["fault_type"]
        if fault_type == fault_type_dict['location fault']:
            total_fault_num += 1
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
            IoU = cal_IoU(box, boxes)
            # max_IoU corresponding score

            max_IoU = torch.max(IoU)
            P_k = dec_dict[image_name][torch.argmax(IoU).item()]["score"]

            # max_IoU corresponding cls_inf
            cls_inf_item = cls_inf_dict[image_name][torch.argmax(IoU).item()]
            cls_max_p = torch.max(torch.tensor(cls_inf_item["full_scores"]))

            assert categorys[torch.argmax(IoU).item()] == cls_inf_item["detectioninf_category_id"], "category not match"

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

    plt_cruve(results, 'location fault', total_fault_num)

    # # read class_indict
    # class_dict = {}
    # json_file = os.path.join('dataset/VOCdevkit/VOC2012', 'pascal_voc_classes.json')
    # assert os.path.exists(json_file), 'json file not found'
    # with open(json_file, 'r') as fp:
    #     class_dict = json.load(fp)
    # # convert class_indict to {class_id:class_name}
    # class_dict = dict((val, key) for key, val in class_dict.items())
    #
    # # save images of top 100 poor localization results
    # for i in range(100):
    #     image_name = results[i]["image_name"]
    #     gtbox = results[i]["gtbox"]
    #     infbox = results[i]["infbox"]
    #     gt_category = results[i]["gt_category"]
    #     cls_gt_category = results[i]["cls_gt_category"]
    #     inf_category = results[i]["inf_category"]
    #     cls_inf_category = results[i]["cls_inf_category"]
    #     poorloc_loss = results[i]["poorloc_loss"]
    #
    #     image = Image.open("dataset/VOCdevkit/VOC2012/JPEGImages/" + image_name).convert("RGB")
    #     gt_category_name = class_dict[gt_category]
    #     cls_gt_category_name = class_dict[cls_gt_category]
    #     inf_category_name = class_dict[inf_category]
    #     cls_inf_category_name = class_dict[cls_inf_category]
    #     image = Image.open("dataset/VOCdevkit/VOC2012/JPEGImages/" + image_name).convert("RGB")
    #     # draw image with gtbox(red) with gt_category_name,cls_gt_category_name and infbox(blue) with inf_category_name,cls_inf_category_name
    #     plt.gca().add_patch(
    #         plt.Rectangle((gtbox[0], gtbox[1]), gtbox[2] - gtbox[0], gtbox[3] - gtbox[1], fill=False,
    #                       edgecolor='red', linewidth=2))
    #     plt.gca().text(gtbox[0], gtbox[1] - 2,
    #                    '{:s} guess: {:s}'.format(gt_category_name, cls_gt_category_name),
    #                    bbox=dict(facecolor='red', alpha=0.5), fontsize=12, color='white')
    #     plt.gca().add_patch(
    #         plt.Rectangle((infbox[0], infbox[1]), infbox[2] - infbox[0], infbox[3] - infbox[1], fill=False,
    #                       edgecolor='blue', linewidth=2))
    #     plt.gca().text(infbox[0], infbox[3] - 2,
    #                    '{:s} guess: {:s}'.format(inf_category_name, cls_inf_category_name),
    #                    bbox=dict(facecolor='blue', alpha=0.5), fontsize=12, color='white')
    #
    #     plt.axis('off')
    #     plt.imshow(image)
    #     plt.savefig("data/detective_results/poorloc/" + str(i) + '_' + image_name, bbox_inches='tight',
    #                 pad_inches=0)
    #     plt.cla()


def Redundancy():
    print("Redundancy")
    cls_gt = []  # ./data/classification/classification_VOCgt_inferences.json
    dec = []  # ./data/detection_results/ssd_VOCval_inferences.json
    with open('./data/classification_results/classification_VOCgtmixedfault_inferences.json', 'r') as f:
        cls_gt = json.load(f)
    with open('./data/detection_results/ssd_VOCval_inferences.json', 'r') as f:
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
    total_fault_num = 0
    fault_type_dict = params.fault_type
    # for every instance in cls_gt
    for i in tqdm(range(len(cls_gt))):
        # image name of this instance
        image_name = cls_gt[i]["image_name"]
        box = cls_gt[i]["bbox"]
        cls_full_score = cls_gt[i]["full_scores"]
        detectiongt_category = cls_gt[i]["detectiongt_category_id"]
        fault_type = cls_gt[i]["fault_type"]
        if fault_type == fault_type_dict["redundancy fault"]:
            total_fault_num += 1
        # if this image is in dec_dict
        if image_name in dec_dict:
            # boxes of this image in dec_dict
            boxes = [j["bbox"] for j in dec_dict[image_name]]
            # IoU of this instance and boxes in dec_dict
            IoU = cal_IoU(box, boxes)
            # if IoU < params.t_b
            if torch.max(IoU) < params.t_b:
                # cal cross_entropy loss of cls_full_score and detectiongt_category
                loss = cross_entropy_loss(cls_full_score, detectiongt_category)

                # save this instance and loss to results
                results.append(
                    {"image_name": image_name, "redundancy_loss": loss, "bbox": box, "full_scores": cls_full_score,
                     "detectiongt_category_id": detectiongt_category, "fault_type": fault_type})

    # sort results by loss from large to small
    results = sorted(results, key=lambda x: x["redundancy_loss"], reverse=True)

    plt_cruve(results, "redundancy fault", total_fault_num)

    # # read class_indict
    # class_dict = {}
    # json_file = os.path.join('dataset/VOCdevkit/VOC2012', 'pascal_voc_classes.json')
    # assert os.path.exists(json_file), 'json file not found'
    # with open(json_file, 'r') as fp:
    #     class_dict = json.load(fp)
    # # convert class_indict to {class_id:class_name}
    # class_dict = dict((val, key) for key, val in class_dict.items())
    #
    # # save imgages of top 100 results with box and class_name1 and class_name2
    # for i in range(100):
    #     image_name = results[i]["image_name"]
    #     box = results[i]["bbox"]
    #     full_scores = results[i]["full_scores"]
    #     detectiongt_category_id = results[i]["detectiongt_category_id"]
    #     loss = results[i]["redundancy_loss"]
    #
    #     image = Image.open("dataset/VOCdevkit/VOC2012/JPEGImages/" + image_name).convert("RGB")
    #
    #     class_name1 = class_dict[detectiongt_category_id]
    #     class_name2 = class_dict[torch.argmax(torch.tensor(full_scores)).item()]
    #
    #     # draw image with bounding box and labels
    #     plt.gca().add_patch(
    #         plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red',
    #                       linewidth=2))
    #     plt.gca().text(box[0], box[1] - 2, '{:s} guess: {:s}{:.3f}'.format(class_name1, class_name2, loss),
    #                    bbox=dict(facecolor='red', alpha=0.5), fontsize=12, color='white')
    #     # 关闭坐标轴 去除图片空白部分
    #     plt.axis('off')
    #     plt.imshow(image)
    #     plt.savefig("data/detective_results/redundancy/" + str(i) + '_' + image_name, bbox_inches='tight', pad_inches=0)
    #     plt.cla()
    #
    # print("results length: ", len(results))


def Missing():
    print("Missing")
    cls_inf = []  # ./data/classification/classification_VOCinf0.5_inferences.json
    cls_gt = []  # ./data/lassification/classification_VOCgt_inferences.json
    with open('./data/classification_results/classification_VOCinf' + str(params.m_t) + '_inferences.json', 'r') as f:
        cls_inf = json.load(f)
    with open('./data/classification_results/classification_VOCgtmixedfault_inferences.json', 'r') as f:
        cls_gt = json.load(f)

    fault_type_dict = parameters().fault_type

    # missing list
    missing_list = []
    with open('./data/fault_annotations/VOCval_mixedfault.json', 'r') as f:
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
            IoU = cal_IoU(box, boxes)
            # if IoU < params.t_f

            if torch.max(IoU) < params.t_f:
                # loss = max(cls_full_score)
                loss = torch.max(torch.tensor(cls_full_score))

                # if this image is in missing_dict
                if image_name in missing_dict:
                    missing_boxes = [j["boxes"] for j in missing_dict[image_name]]
                    # max IoU of this box and missing boxes
                    missing_max_IoU = torch.max(cal_IoU(box, missing_boxes))
                    if missing_max_IoU > 0.5:
                        fault_type = fault_type_dict['missing fault']

                # save this instance and loss to results
                results.append(
                    {"image_name": image_name, "missing_loss": loss, "bbox": box, "full_scores": cls_full_score,
                     "detectioninf_category_id": detectioninf_category, "fault_type": fault_type})

    # sort results by loss from large to small
    results = sorted(results, key=lambda x: x["missing_loss"], reverse=True)
    plt_cruve(results, "missing fault", int(15787 * params.fault_ratio))


# 15787
# # read class_indict
# class_dict = {}
# json_file = os.path.join('dataset/VOCdevkit/VOC2012', 'pascal_voc_classes.json')
# assert os.path.exists(json_file), 'json file not found'
# with open(json_file, 'r') as fp:
#     class_dict = json.load(fp)
# # convert class_indict to {class_id:class_name}
# class_dict = dict((val, key) for key, val in class_dict.items())
#
# # save imgages of top 100 results with box and class_name1 and class_name2
# for i in range(100):
#     image_name = results[i]["image_name"]
#     box = results[i]["bbox"]
#     full_scores = results[i]["full_scores"]
#     detectiongt_category_id = results[i]["detectioninf_category_id"]
#     loss = results[i]["missing_loss"]
#
#     image = Image.open("dataset/VOCdevkit/VOC2012/JPEGImages/" + image_name).convert("RGB")
#
#     class_name1 = class_dict[detectiongt_category_id]
#     class_name2 = class_dict[torch.argmax(torch.tensor(full_scores)).item()]
#
#     # draw image with bounding box and labels
#     plt.gca().add_patch(
#         plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='blue',
#                       linewidth=2))
#     plt.gca().text(box[0], box[1] - 2, '{:s} guess: {:s}{:.3f}'.format(class_name1, class_name2, loss),
#                    bbox=dict(facecolor='blue', alpha=0.5), fontsize=12, color='white')
#
#     # draw image's all gt box
#     for j in range(len(cls_gt_dict[image_name])):
#         box = cls_gt_dict[image_name][j]["bbox"]
#         class_id = cls_gt_dict[image_name][j]["detectiongt_category_id"]
#         class_name = class_dict[class_id]
#         plt.gca().add_patch(
#             plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red',
#                           linewidth=2))
#         plt.gca().text(box[0], box[1] - 2, '{:s}'.format(class_name),
#                        bbox=dict(facecolor='red', alpha=0.5), fontsize=12, color='white')
#
#     # 关闭坐标轴 去除图片空白部分
#     plt.axis('off')
#     plt.imshow(image)
#     plt.savefig("data/detective_results/missing/" + str(i) + '_' + image_name, bbox_inches='tight', pad_inches=0)
#     plt.cla()
#
# print("results length: ", len(results))


if __name__ == "__main__":
    Missing()

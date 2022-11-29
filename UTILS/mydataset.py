import json
import os

import torch
from PIL import Image
from lxml import etree

import torch.utils.data as data
from matplotlib import pyplot as plt


class VOCclassificationDataSet(data.Dataset):
    def __init__(self, voc_root, transforms=None, txt_name: str = "train.txt"):
        self.root = voc_root
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")
        self.transforms = transforms
        text_path = os.path.join(self.root, "ImageSets/Main", txt_name)
        assert os.path.exists(text_path), "file not found"
        with open(text_path, "r") as f:
            xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                        for line in f.readlines() if len(line.strip()) > 0]
        self.xml_list = []
        # check file
        for xml_path in xml_list:
            assert os.path.exists(xml_path), "xml file not found"
            with open(xml_path, "r") as f:
                xml_str = f.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            if "object" not in data:
                print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                continue
            self.xml_list.append(xml_path)
        # read class_indict
        json_file = os.path.join(self.root, 'pascal_voc_classes.json')
        assert os.path.exists(json_file), 'json file not found'
        with open(json_file, 'r') as fp:
            self.class_dict = json.load(fp)

        # split into instances
        self.instances_list = []
        for xml_path in self.xml_list:
            with open(xml_path, "r") as f:
                xml_str = f.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            for obj in data["object"]:
                instance = {}
                instance["image_id"] = data["filename"]
                instance["category_id"] = self.class_dict[obj["name"]]
                instance["bbox"] = obj["bndbox"]
                self.instances_list.append(instance)
        print(f"INFO: {len(self.instances_list)} instances loaded.")

    def __getitem__(self, idx):
        instance = self.instances_list[idx]
        img_path = os.path.join(self.img_root, instance["image_id"])
        img = Image.open(img_path).convert("RGB")
        boxes = [int(instance["bbox"]["xmin"]), int(instance["bbox"]["ymin"]),
                 int(instance["bbox"]["xmax"]), int(instance["bbox"]["ymax"])]
        label = instance["category_id"]

        # # # draw img with bounding box
        # plt.gca().add_patch(
        #     plt.Rectangle((boxes[0], boxes[1]), boxes[2] - boxes[0], boxes[3] - boxes[1], fill=False, edgecolor='red',
        #                   linewidth=3)) # xmin, ymin, w, h
        # # plt original image
        # plt.imshow(img)
        # plt.show()

        # Crop out the boxes part of the image
        img = img.crop(boxes)

        # # plt cropped image
        # plt.imshow(img)
        # plt.show()

        # Resize the image to 224x224
        img = img.resize((224, 224))

        # # plt resized image
        # plt.imshow(img)
        # plt.show()

        # convert everything into a torch.Tensor
        if self.transforms is not None:
            img = self.transforms(img)
        label = torch.tensor(label)

        return img, label

    def __len__(self):
        return len(self.instances_list)

    def parse_xml_to_dict(self, xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    # collate_fn needs for batch
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


class inference_VOCGt_classificationDataSet(data.Dataset):
    def __init__(self, voc_root, transforms=None, txt_name: str = "train.txt"):
        self.root = voc_root
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")
        self.transforms = transforms
        text_path = os.path.join(self.root, "ImageSets/Main", txt_name)
        assert os.path.exists(text_path), "file not found"
        with open(text_path, "r") as f:
            xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                        for line in f.readlines() if len(line.strip()) > 0]
        self.xml_list = []
        # check file
        for xml_path in xml_list:
            assert os.path.exists(xml_path), "xml file not found"
            with open(xml_path, "r") as f:
                xml_str = f.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            if "object" not in data:
                print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                continue
            self.xml_list.append(xml_path)
        # read class_indict
        json_file = os.path.join(self.root, 'pascal_voc_classes.json')
        assert os.path.exists(json_file), 'json file not found'
        with open(json_file, 'r') as fp:
            self.class_dict = json.load(fp)

        # split into instances
        self.instances_list = []
        for xml_path in self.xml_list:
            with open(xml_path, "r") as f:
                xml_str = f.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            for obj in data["object"]:
                instance = {}
                instance["image_name"] = data["filename"]
                instance["category_id"] = self.class_dict[obj["name"]]
                instance["bbox"] = obj["bndbox"]
                self.instances_list.append(instance)
        print(f"INFO: {len(self.instances_list)} instances loaded.")

    def __getitem__(self, idx):
        instance = self.instances_list[idx]
        img_path = os.path.join(self.img_root, instance["image_name"])
        img = Image.open(img_path).convert("RGB")
        boxes = [int(instance["bbox"]["xmin"]), int(instance["bbox"]["ymin"]),
                 int(instance["bbox"]["xmax"]), int(instance["bbox"]["ymax"])]

        target = {}
        target["image_name"] = instance["image_name"]
        target["category_id"] = torch.tensor(instance["category_id"])
        target["boxes"] = torch.tensor(boxes)

        # # # draw img with bounding box
        # plt.gca().add_patch(
        #     plt.Rectangle((boxes[0], boxes[1]), boxes[2] - boxes[0], boxes[3] - boxes[1], fill=False, edgecolor='red',
        #                   linewidth=3)) # xmin, ymin, w, h
        # # plt original image
        # plt.imshow(img)
        # plt.show()

        # Crop out the boxes part of the image
        img = img.crop(boxes)

        # # plt cropped image
        # plt.imshow(img)
        # plt.show()

        # Resize the image to 224x224
        img = img.resize((224, 224))

        # # plt resized image
        # plt.imshow(img)
        # plt.show()

        # convert everything into a torch.Tensor
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.instances_list)

    def parse_xml_to_dict(self, xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    # collate_fn needs for batch
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


from UTILS.parameters import parameters


class inference_VOCinf_classificationDataSet(data.Dataset):
    def __init__(self, voc_root, inferences_root="../data/ssd_VOCval_infreneces.json", transforms=None):
        self.root = voc_root
        self.img_root = os.path.join(self.root, "JPEGImages")
        full_inference_results = json.load(open(inferences_root, "r"))

        self.inference_results = []
        # get inference results score > m_t
        params = parameters()
        for inference_result in full_inference_results:
            if inference_result["score"] > params.m_t:
                self.inference_results.append(inference_result)
        print(f"INFO: {len(self.inference_results)} instances loaded.")

        self.transforms = transforms


    def __getitem__(self, idx):
        instance = self.inference_results[idx]
        img_path = os.path.join(self.img_root, instance["image_name"])
        img = Image.open(img_path).convert("RGB")
        boxes = instance["bbox"]

        target = {}
        target["image_name"] = instance["image_name"]
        target["category_id"] = torch.tensor(instance["category_id"])
        target["boxes"] = torch.tensor(boxes)

        # # draw img with bounding box
        # plt.gca().add_patch(
        #     plt.Rectangle((boxes[0], boxes[1]), boxes[2] - boxes[0], boxes[3] - boxes[1], fill=False, edgecolor='red',
        #                   linewidth=3)) # xmin, ymin, w, h
        # # plt original image
        # plt.imshow(img)
        # plt.show()

        # Crop out the boxes part of the image
        img = img.crop(boxes)

        # plt cropped image
        # plt.imshow(img)
        # plt.show()

        # Resize the image to 224x224
        img = img.resize((224, 224))

        # plt resized image
        # plt.imshow(img)
        # plt.show()

        # convert everything into a torch.Tensor
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


    def __len__(self):
        return len(self.inference_results)

    def parse_xml_to_dict(self, xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    # collate_fn needs for batch
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


class VOCDetectionDataSet(data.Dataset):
    def __init__(self, voc_root, transforms=None, txt_name: str = "train.txt"):
        self.root = voc_root
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")
        self.transforms = transforms
        text_path = os.path.join(self.root, "ImageSets/Main", txt_name)
        assert os.path.exists(text_path), "file not found"
        with open(text_path, "r") as f:
            xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                        for line in f.readlines() if len(line.strip()) > 0]
        self.xml_list = []
        # check file
        for xml_path in xml_list:
            assert os.path.exists(xml_path), "xml file not found"
            with open(xml_path, "r") as f:
                xml_str = f.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            if "object" not in data:
                print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                continue
            self.xml_list.append(xml_path)
        # read class_indict
        json_file = os.path.join(self.root, 'pascal_voc_classes.json')
        assert os.path.exists(json_file), 'json file not found'
        with open(json_file, 'r') as fp:
            self.class_dict = json.load(fp)

    def __getitem__(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path, "r") as f:
            xml_str = f.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        img = Image.open(img_path).convert("RGB")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            boxes.append([int(obj["bndbox"]["xmin"]), int(obj["bndbox"]["ymin"]),
                          int(obj["bndbox"]["xmax"]), int(obj["bndbox"]["ymax"])])
            labels.append(self.class_dict[obj["name"]])

            # check if the boxes are valid
            if boxes[-1][2] <= boxes[-1][0] or boxes[-1][3] <= boxes[-1][1]:
                print(f"INFO: invalid box in {xml_path}, skip this annotation file.")
                continue

            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # # # draw img with bounding box and labels
        # for i in range(len(boxes)):
        #     plt.gca().add_patch(
        #         plt.Rectangle((boxes[i][0], boxes[i][1]), boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1], fill=False, edgecolor='red',
        #                       linewidth=3))
        #     # reverse class_dict to get class name
        #     class_name = list(self.class_dict.keys())[list(self.class_dict.values()).index(labels[i])]
        #     plt.gca().text(boxes[i][0], boxes[i][1], '{:s}'.format(class_name), bbox=dict(facecolor='blue', alpha=0.5),
        #                     fontsize=14, color='white')
        # # plt original image
        # plt.imshow(img)
        # plt.show()

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
                target["boxes"][:, 2] - target["boxes"][:, 0])
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        # print(target)
        # convert everything into a torch.Tensor
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.xml_list)

    def parse_xml_to_dict(self, xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    # collate_fn needs for batch
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


class inferenceVOCDetectionDataSet(data.Dataset):
    def __init__(self, voc_root, transforms=None, txt_name: str = "train.txt"):
        self.root = voc_root
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")
        self.transforms = transforms
        text_path = os.path.join(self.root, "ImageSets/Main", txt_name)
        assert os.path.exists(text_path), "file not found"
        with open(text_path, "r") as f:
            xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                        for line in f.readlines() if len(line.strip()) > 0]
        self.xml_list = []
        # check file
        for xml_path in xml_list:
            assert os.path.exists(xml_path), "xml file not found"
            with open(xml_path, "r") as f:
                xml_str = f.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            if "object" not in data:
                print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                continue
            self.xml_list.append(xml_path)
        # read class_indict
        json_file = os.path.join(self.root, 'pascal_voc_classes.json')
        assert os.path.exists(json_file), 'json file not found'
        with open(json_file, 'r') as fp:
            self.class_dict = json.load(fp)

    def __getitem__(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path, "r") as f:
            xml_str = f.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        img = Image.open(img_path).convert("RGB")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            boxes.append([int(obj["bndbox"]["xmin"]), int(obj["bndbox"]["ymin"]),
                          int(obj["bndbox"]["xmax"]), int(obj["bndbox"]["ymax"])])
            labels.append(self.class_dict[obj["name"]])

            # check if the boxes are valid
            if boxes[-1][2] <= boxes[-1][0] or boxes[-1][3] <= boxes[-1][1]:
                print(f"INFO: invalid box in {xml_path}, skip this annotation file.")
                continue

            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # # # draw img with bounding box and labels
        # for i in range(len(boxes)):
        #     plt.gca().add_patch(
        #         plt.Rectangle((boxes[i][0], boxes[i][1]), boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1], fill=False, edgecolor='red',
        #                       linewidth=3))
        #     # reverse class_dict to get class name
        #     class_name = list(self.class_dict.keys())[list(self.class_dict.values()).index(labels[i])]
        #     plt.gca().text(boxes[i][0], boxes[i][1], '{:s}'.format(class_name), bbox=dict(facecolor='blue', alpha=0.5),
        #                     fontsize=14, color='white')
        # # plt original image
        # plt.imshow(img)
        # plt.show()

        target = {}
        target["image_name"] = data["filename"]
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
                target["boxes"][:, 2] - target["boxes"][:, 0])
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        # print(target)
        # convert everything into a torch.Tensor
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.xml_list)

    def parse_xml_to_dict(self, xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    # collate_fn needs for batch
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


# test
if __name__ == "__main__":
    # test inference_VOCinf_classificationDataSet
    dataset = inference_VOCinf_classificationDataSet(voc_root="../dataset/VOCdevkit/VOC2012", transforms=None,
                                                     inferences_root="../data/ssd_VOCval_inferences.json")
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    # test dataloader for 10 images
    for i, (img, target) in enumerate(dataloader):
        if i == 10:
            break
        print(target)

    #     # test VOCDetectionDataSet
    # dataset = inferenceVOCDetectionDataSet(voc_root="../dataset/VOCdevkit/VOC2012/", txt_name="train.txt")
    # dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    # # test dataloader for 10 images
    # for i, (img, target) in enumerate(dataloader):
    #     print(target)
    #     if i == 10:
    #         break

    # from torchvision import transforms
    # from torch.utils.data import DataLoader
    #
    # data_transforms = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    #
    # dataset = VOCclassificationDataSet(voc_root="./dataset/VOCdevkit/VOC2012", transforms=data_transforms, txt_name="train.txt")
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
    #
    #
    # # test dataloader for 10 images
    # for i, (img, label) in enumerate(dataloader):
    #     if i == 10:
    #         break
    #     print(f"img shape: {img[0].shape}, label: {label[0]}")

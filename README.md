# Datactive

This repository is an implementation of the paper: Datactive: Dataset Debugging for Object Detection Systems.

Object detection models serve as a crucial componet in many intelligent software systems, and the accuracy of
such models depends on the quality of the training dataset.
However, it was observed that training data usually contains
various errors due to the error-prone annotation procedure,
and such errors can significantly influence the performance of
object detection models, and thus propagate effect to the entire
system. Each object detection input may contain multiple objects
of different categories that need to be detected, and there may
also be spatial relationships and occlusions between them. Thus
the core challenge in annotating object detection data lies in
identifying all objects that require annotation and correctly
bounding and classifying them.

In this paper, we propose a universal method that does not
rely on object detection models’ predictions, namely Datactive.
Based on the fundamental idea of data decoupling and reorgani-
zation, Datactive constructs debugging models by separating
existing foreground annotations and comprehensive annotations
containing background information. A robust training strategy
is adopted so that Datactive can be applied to both previously
seen and unseen datasets. The debugging priority scores are
assigned to both existing bounding boxes and background areas
based on debug model results. This helps testers quickly identify
which annotations need to be re-examined in the given input or
dataset. To verify the effectiveness of our approach, we conduct
experiments on three benchmark datasets of diverse application
scenarios. Object detection prediction-based methods are applied
as baselines to compare both the effectiveness and diversity,
demonstrating the practical value of our approach.
![overview](./pictures/overview.Png) 

## Installation
`pip install -r requirements.txt`

## Usage
You should first transform the annotation of the object detection dataset into the following format `.json` file:
```python
[
    # an instance
    {
        "image_name": "000000312552.jpg",
        "image_size": [
            400,
            300
        ],
        "boxes": [ #x1,y1,x2,y2
            165.86,
            129.96,
            207.41000000000003,
            161.83
        ],
        "labels": 1,
        "image_id": 312552,
        "area": 798.7644499999996,
        "iscrowd": 0,
        "fault_type": 0
    },
  ...
]
```

Then you can run the following command to debug the dataset:
+ `python demo.py --dataset ./dataset/COCO --trainlabel ./dataset/COCO/trainlabel.json --testlabel ./dataset/COCO/testlabel.json --classnum 80`

Parameter explanation:

`--dataset` : Location of dataset image storage.

`--trainlabel` : The above transformed trainlabel.

`--testlabel` : The above transformed testlabel.

`--classnum` : Number of categories in the dataset.

## Results
Some `demo.py` results on COCO are shown below :
<div><table frame=void>	<!--用了<div>进行封装-->
	<tr>
        <td><div><center>	<!--每个格子内是图片加标题-->
        	<img src="./pictures/35.png"
                 height="120"/>	<!--高度设置-->
        	<br>	<!--换行-->
        </center></div></td>    
     	<td><div><center>	<!--第二张图片-->
    		<img src="./pictures/67.png"
                 height="120"/>	
    		<br>
        </center></div></td>
        <td><div><center>	<!--每个格子内是图片加标题-->
        	<img src="./pictures/111.png"
                 height="120"/>	<!--高度设置-->
        	<br>	<!--换行-->
        </center></div></td> 
        <td><div><center>	<!--每个格子内是图片加标题-->
        	<img src="./pictures/123.png"
                 height="120"/>	<!--高度设置-->
        	<br>	<!--换行-->
        </center></div></td> 
        <td><div><center>	<!--每个格子内是图片加标题-->
        	<img src="./pictures/141.png"
                 height="120"/>	<!--高度设置-->
        	<br>	<!--换行-->
        </center></div></td> 
	</tr>
</table></div>

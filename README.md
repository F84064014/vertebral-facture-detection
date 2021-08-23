# Vertebral Facture Detection Based On Machine Learning Method

Our aim is to detect if each vertebra in a given X-ray image is fractured, here we try to identify if *Thoracic* and *Lumbar* is fractured by ML/DL model. The data we used for training model is provided by National Cheng Kung University Hospital, Taiwan, Tainan.

# Description

In this project we will use both machine learning and deep learning method. Deep learning method is mainly for detecting and segmentation of vertebra, then we will extract some critical information form the result of segmentation, these information are taken to be the feature for our machine learning method to identify if the vertebra is fractured, screwed or normal.

# Process Flow

1. Labeling Data <br/><br/>
manual labeling data with twelve-sided polygon for each vertebra. <br/><br/>
each vetebra labeled with its position (T1-T12, L1-L5, S) <br/><br/>
![image1](./image/manual_labeling.png)
2. Object Detection for Vertebra <br/><br/>
![image1](./image/DetectionDemo.jpg)
3. Image segmentation for Vertebra <br/><br/>
![image1](./image/unetdemo.png)
4. Feature Extraction <br/><br/>
![image1](./image/feat6.png)
5. Data Analysis <br/><br/>
![image1](./image/rl_scatter.jpg)
6. Detection <br/><br/>
for each vertebra detected, extract feature and predict the result (normal, screw or fractured)

# Demo Image

there are three color blue, green and red represent normal, screwed and fractured respectively.<br/><br/>
the following examples are from RandomForest algorithm
- 01549476_FILE5.jpg<br/><br/>
![image1](./image/01549476_FILE5.jpg)<br/><br/>
- 01990231_FILE3.jpg<br/><br/>
![image2](./image/01990231_FILE3.jpg)<br/><br/>
- 03877078_FILE2.jpg<br/><br/>
![image3](./image/03877078_FILE2.jpg)<br/><br/>

# A Deep Local and Global Scene Graph Matching for Image-Text Retrieval

This is the repository of the **Multimodal scene graph matching for cheapfakes detection** which is under review in ICMR2024. This research is inspired by the SGM and LGSGM [paper](https://arxiv.org/abs/2106.02400).

Our code is mostly based on the LGSGM original [code](https://github.com/m2man/LGSGM).


## 1. Requirements
Please install packages in the ```requirements.txt```. The project is implemented with python 3.7.9

## 2. Data prepare
Our data (Flickr30k) is original given by the SGM [paper](https://arxiv.org/abs/1910.05134). We only performed same basic cleaning process to remove duplicated data and lowering text. The preprocessed data can be found in the Data folder.

The model also need the visual features which are the embedded vector of objects in images. In this research, we used EfficientNet-b5 to extract the features. You can extract by running ```extract_visual_features.py``` script. We also uploaded our prepared features ([here](https://drive.google.com/drive/folders/1IvlmTZ9wUpOVIr9MzPgWZB5aYTaTD0jn?usp=sharing)). You can download it and place in the **Data folder**.

## 3. Training and Evaluating
You can run the ```main_train.py``` script to perform either training or evaluating the model. Our pretrained model can be found [here](https://drive.google.com/drive/folders/100t_GxbhycwfQO82cz-7Xfkn8_t69_Vz?usp=sharing). Please download it and place in **Report folder**.

## 4. Contact
For any issue or comment, you can directly email me at tamnm1032@gmail.com

For citation, you can add the bibtex as following:

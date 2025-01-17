# Multimodal Scene Graph Matching for Cheapfakes Detection

This is the repository of the **Multimodal scene graph matching for cheapfakes detection** which is under review in ICMR2024. This research is inspired by the SGM and LGSGM [paper](https://arxiv.org/abs/2106.02400).

Our code is mostly based on the LGSGM original [code](https://github.com/m2man/LGSGM).


## 1. Requirements
Please install packages in the ```requirements.txt```. The project is implemented with python 3.7.9

## 2. Data prepare
Our data is original given by the ICME2023 Grand Challenge on Cheapfakes detection. You can fill out the [form](https://forms.gle/jj7jLhF4b43KKxLZ7) to have access to the dataset.

The model also need the visual features which are the embedded vector of objects in images. In this research, we used EfficientNet-b5 to extract the features. You can extract by running ```extract_visual_features.py``` script. We also uploaded our prepared features [here](https://www.dropbox.com/scl/fo/b7vca83ei0mcqomdtijob/h?rlkey=f7h2470ezq6vz1xfict043d41&dl=0). You can download it and place in the **Data folder**.

## 3. Training and Evaluating
You can run the ```main_train.py``` script to perform either training or evaluating the model. Our pretrained model can be found [here](https://drive.google.com/drive/folders/100t_GxbhycwfQO82cz-7Xfkn8_t69_Vz?usp=sharing). Please download it and place in **Report folder**. To run the second approach using the cosine similarities, you can access the files in the ``Cossim_approach`` folder.

## 4. Contact
For any issue or comment, you can directly email me at tamnm1032@gmail.com

For citation, you can add the bibtex as following:

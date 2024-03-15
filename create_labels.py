import pandas as pd
import json
import random
from tqdm import tqdm
from PIL import Image
import numpy as np
from simclr import SimCLR
from torchvision.models import resnet50
import torchvision
import torch.nn as nn
import torch
import random
import re

f = open('mmsys_anns/train_data.json')
json_content = f.read()
json_strings = json_content.split('\n')
data = [json.loads(js) for js in json_strings if js]

f = open('similar_images_train_filter.json')
json_content = f.read()
duplicate_images = json.loads(json_content)

def count_captions(sample):
    return len(sample['articles'])
def check_duplicate(sample, duplicate_list):
    img = sample['img_local_path'].split('/')[-1]
    if img in duplicate_list:
        return False
    return True

list_25000_images = random.sample([sample for sample in data if check_duplicate(sample,duplicate_images) and count_captions(sample)>2], k = 25000)

#Create Not-out-of-context (NOOC) labels
dict_samples_0 = []
list_20000_images = random.sample(list_25000_images, k=20000)
for sample in tqdm(list_20000_images):
    dict_sample_1 = {}
    dict_sample_2 = {}
    dict_sample_1['img_local_path'] = sample['img_local_path']
    dict_sample_2['img_local_path'] = sample['img_local_path']
    captions = random.choices(sample['articles'], k=3)
    #captions = sample['articles'][:2]
    dict_sample_1['caption1'] = captions[0]['caption']
    dict_sample_1['caption1_modified'] = captions[0]['caption_modified']
    dict_sample_1["caption1_entities"] = captions[0]['entity_list']
    dict_sample_1['caption2'] = captions[1]['caption']
    dict_sample_1['caption2_modified'] = captions[1]['caption_modified']
    dict_sample_1["caption2_entities"] = captions[1]['entity_list']
    dict_sample_1['context_label'] = 0
    dict_sample_2['caption1'] = captions[0]['caption']
    dict_sample_2['caption1_modified'] = captions[0]['caption_modified']
    dict_sample_2["caption1_entities"] = captions[0]['entity_list']
    dict_sample_2['caption2'] = captions[2]['caption']
    dict_sample_2['caption2_modified'] = captions[2]['caption_modified']
    dict_sample_2["caption2_entities"] = captions[2]['entity_list']
    dict_sample_2['context_label'] = 0
    dict_samples_0.append(dict_sample_1)
    dict_samples_0.append(dict_sample_2)
    
    
#Create unmatch captions
list_20000_images_1 = random.sample(list_25000_images, k = 20000)
device = torch.device('cuda:0')

encoder = resnet50(pretrained=False)
projection_dim = 64
n_features = encoder.fc.in_features  # get dimensions of last fully-connected layer
model = SimCLR(encoder, projection_dim, n_features)
model.load_state_dict(torch.load("checkpoint_100.tar"))
model = model.to(device)
model.eval()

preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        )

cos = nn.CosineSimilarity()#dim=0

def encode(file_path, transform=preprocess, model=model, device=device):
    img = Image.open(file_path).convert("RGB")
    img_pre = transform(img)
    img_batch = torch.unsqueeze(img_pre, 0).to(device)
    img_encode = model.encoder(img_batch)
    return img_encode
    
#check if two images are similar or not
def check_duplicate_images(sample_1, sample_2, transform=preprocess, cosine=cos, model=model, device=device): 
    img_1 = Image.open(sample_1['img_local_path']).convert("RGB")
    img_1_pre = transform(img_1)
    img_2 = Image.open(sample_2['img_local_path']).convert("RGB")
    img_2_pre = transform(img_2)
    batch = torch.stack([img_1_pre, img_2_pre]).to(device)
    vector_1, vector_2 = model.encoder(batch)
    cosine_similarity = cosine(vector_1,vector_2).item()
    if cosine_similarity > 0.6:
        return True
    return False
    
dict_samples_1 = []
for sample in tqdm(list_20000_images_1):
    dict_sample = {}
    dict_sample['img_local_path'] = sample['img_local_path']
    captions = random.sample(sample['articles'], 1)
    dict_sample['caption1'] = captions[0]['caption']
    dict_sample['caption1_modified'] = captions[0]['caption_modified']
    dict_sample["caption1_entities"] = captions[0]['entity_list']
    sample_2 = random.sample(list_20000_images_1, 1)[0]
    vector_img = encode(sample['img_local_path'])
    vector_check = encode(sample_2['img_local_path'])
    while cos(vector_img, vector_check).item() > 0.6:
        sample_2 = random.sample(list_20000_images_1, 1)[0]
        vector_check = encode(sample_2['img_local_path'])
    captions_2 = random.sample(sample_2['articles'], 1)
    dict_sample['caption2'] = captions_2[0]['caption']
    dict_sample['caption2_modified'] = captions_2[0]['caption_modified']
    dict_sample["caption2_entities"] = captions_2[0]['entity_list']
    dict_sample['context_label'] = 1
    dict_samples_1.append(dict_sample)
   
# Changing entities in the captions
def extract_entities(data):
    entities = {"PERSON": set(), "GPE": set(), "DATE": set()}
    count = 0
    for entity in data["entity_list"]:
        entity_type = entity[1]
        if entity_type in entities:
            entities[entity_type].add(entity[0])
            count += 1
    return entities, count

def merge_two_entities(entities_1, entities_2):
    entities = {"PERSON": set(), "GPE": set(), "DATE": set()}
    entities["PERSON"] = entities_1["PERSON"].union(entities_2["PERSON"])
    entities["GPE"] = entities_1["GPE"].union(entities_2["GPE"])
    entities["DATE"] = entities_1["DATE"].union(entities_2["DATE"])
    return entities

def extract_entities_image(data):
    entities = {"PERSON": set(), "GPE": set(), "DATE": set()}
    for article in data["articles"]:
        temp_entities, count = extract_entities(article)
        entities = merge_two_entities(entities, temp_entities)
    return entities

def create_list_entities(data):
    entities = {"PERSON": set(), "GPE": set(), "DATE": set()}
    for sample in data:
        temp_entities = extract_entities_image(sample)
        entities = merge_two_entities(entities, temp_entities)
    return entities
    
X1 = create_list_entities(data)

def replace_phrase(sentence, target_phrase, type, entity_list):
    #target_words = target_phrase.split()
    #random_phrase = ' '.join(random_word(word) for word in target_words)
    random_phrase = random_word(target_phrase, type)
    #modified_sentence = re.sub(re.escape(target_phrase), random_phrase, sentence)
    modified_sentence = sentence.replace(target_phrase, random_phrase)
    entities = [[random_phrase, entity[1]] if entity[0]==target_phrase else entity for entity in entity_list]
    return modified_sentence, entities

def random_word(word, type):
    random_entity = random.choice(tuple(X1[type]))
    while random_entity == word:
        random_entity = random.choice(tuple(X1[type]))
    return random_entity
    
list_10000_images_1 = random.sample(list_25000_images, k=10000)
dict_samples_2 = []
for sample in tqdm(list_10000_images_1):
    dict_sample = {}
    dict_sample['img_local_path'] = sample['img_local_path']
    caption = random.sample(sample['articles'], 1)[0]
    dict_sample['caption1'] = caption['caption']
    dict_sample['caption1_modified'] = caption['caption_modified']
    dict_sample["caption1_entities"] = caption['entity_list']
    entities, count = extract_entities(caption)
    if count < 2:
        continue
    modified_caption = caption['caption']
    entity_list = caption['entity_list'].copy()
    for entity_type, entity_set in entities.items():
        for entity in entity_set:
            if entity in caption['caption']:
                try:
                    modified_caption, entity_list = replace_phrase(modified_caption, entity, entity_type, entity_list)
                except:
                    print(caption)
                    modified_caption, entity_list = replace_phrase(modified_caption, entity, entity_type, entity_list)
    dict_sample['caption2'] = modified_caption
    dict_sample['caption2_modified'] = caption['caption_modified']
    dict_sample["caption2_entities"] = entity_list
    dict_sample['context_label'] = 1
    dict_samples_2.append(dict_sample_temp_1)
    
# Create distraction phrases
def create_notification_captions(caption):
    types = ["front", "back"]
    sentences_1 = ["This is fake news", "This is false information", "The news is wrong", "This hoax actually fooled lots of people", "The information is fabricated", "This is unfounded rumours", "This is fake news"]
    sentences_2 = ["This is groundless conspiracy theories", "There is no evidence", "This splicing created a false impression", "This is a fabricated one", "It unwittingly fooled thousands of people"]
    sentences = sentences_1 + sentences_2
    type = random.choice(types)
    sentence = random.choice(sentences)
    if type == "front":
        new_caption = sentence + ". " + caption
    elif type == "back":
        if caption.endswith("."):
            caption = caption[:-1]
        new_caption = caption + ". " + sentence + "."
    return new_caption
    
dict_samples_3 = []
list_10000_images_2 = random.sample(list_25000_images, k=10000)
for sample in tqdm(list_10000_images_2):
    dict_sample = {}
    dict_sample['img_local_path'] = sample['img_local_path']
    #captions = random.sample(sample['articles'], 2)
    captions = sample['articles']
    dict_sample['caption1'] = captions[0]['caption']
    dict_sample['caption1_modified'] = captions[0]['caption_modified']
    dict_sample["caption1_entities"] = captions[0]['entity_list']
    modified_caption = create_notification_captions(captions[1]['caption'])
    dict_sample['caption2'] = modified_caption
    dict_sample['caption2_modified'] = captions[1]['caption_modified']
    dict_sample["caption2_entities"] = captions[1]['entity_list']
    dict_sample['context_label'] = 1
    dict_samples_3.append(dict_sample_temp_2)

# Combine and save the results
final_dict = dict_samples_0 + dict_samples_1 + dict_samples_2 + dict_samples_3
with open('cheapfake_80000_train.json', 'w') as file:
    for item in final_final_dict:
        json_line = json.dumps(item)
        file.write(json_line + '\n')
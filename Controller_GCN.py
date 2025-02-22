# Precompute EfficientNet before, not run EfficientNet here
# Include the Predicate visual Ft
# Add Extra GCN for textual graph (after the RNN)
from data_utils import *
import models as md
from metrics import *
from torch.nn import BCELoss
from retrieval_utils import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import itertools
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.nn.utils.clip_grad import clip_grad_norm_ as clip_grad_norm
import time
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import os
import torchvision.transforms as transforms
import numpy as np
from transformers import AutoTokenizer


device = torch.device('cuda:0')
#device = torch.device('cpu')
non_blocking = True
# device = torch.cuda.set_device(0)
    
DATA_DIR = './Data'

#word2idx_cap = joblib.load(f"../NewData/flickr30k_caps_word2idx.joblib") # This dictionary include the above
word2idx_cap = joblib.load(f"{DATA_DIR}/train/cheapfake_lowered_caps_word2idx_train_80000.joblib")
word2idx_img_obj = joblib.load(f"{DATA_DIR}/flickr30k_lowered_img_obj_word2idx.joblib") 
word2idx_img_pred = joblib.load(f"{DATA_DIR}/flickr30k_lowered_img_pred_word2idx.joblib") 


TOTAL_CAP_WORDS = len(word2idx_cap)
TOTAL_IMG_OBJ = len(word2idx_img_obj)
TOTAL_IMG_PRED = len(word2idx_img_pred)

# lemmatized or lowered
#images_data_train = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_images_data.joblib")
#caps_data_train = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_caps_data.joblib")
#images_data_train = joblib.load(f"{DATA_DIR}/visual_test_100_images.joblib")
#caps_data_train = joblib.load(f"{DATA_DIR}/caption_test_100_images.joblib")

subset = 'test'
#images_data_val = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_images_data.joblib")
#caps_data_val = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_caps_data.joblib")
#images_data_val = joblib.load(f"{DATA_DIR}/visual_test_100_images.joblib")
#caps_data_val = joblib.load(f"{DATA_DIR}/caption_test_100_images.joblib")
images_data_val = joblib.load(f"{DATA_DIR}/{subset}/cheapfake_{subset}_lowered_images_data_Pe-NET.joblib")
caps_data_val = joblib.load(f"{DATA_DIR}/{subset}/cheapfake_{subset}_lowered_caps_data.joblib")
df_val = pd.read_csv(f"{DATA_DIR}/{subset}/label_file_{subset}.csv")
OBJ_FT_DIR_val = f'{DATA_DIR}/{subset}/VisualObjectFeatures' # run extract_visual_features.py to get this
PRED_FT_DIR_val = f'{DATA_DIR}/{subset}/VisualPredFeatures' # run extract_visual_features.py to get this

init_embed_model_weight_cap = joblib.load(f'{DATA_DIR}/train/init_glove_embedding_weight_lowered_train_80000.joblib')
init_embed_model_weight_cap = torch.FloatTensor(init_embed_model_weight_cap)
init_embed_model_weight_img_obj = joblib.load(f'{DATA_DIR}/init_glove_embedding_weight_lowered_img_obj.joblib')
init_embed_model_weight_img_obj = torch.FloatTensor(init_embed_model_weight_img_obj)
init_embed_model_weight_img_pred = joblib.load(f'{DATA_DIR}/init_glove_embedding_weight_lowered_img_pred.joblib')
init_embed_model_weight_img_pred = torch.FloatTensor(init_embed_model_weight_img_pred)

def print_dict(di):
    result = ''
    for key, val in di.items():
        key_upper = key.upper()
        result += f"{key_upper}: {val}\n"
    return result
    
class Trainer():
    def __init__(self, info_dict):
        super(Trainer, self).__init__()
        ##### INIT #####
        self.info_dict = info_dict
        self.info_dict['total_img_obj'] = TOTAL_IMG_OBJ
        self.info_dict['total_img_pred'] = TOTAL_IMG_PRED
        self.info_dict['total_cap_words'] = TOTAL_CAP_WORDS
        self.numb_sample = info_dict['numb_sample'] # 50000 - number of training sample in 1 epoch
        self.numb_epoch = info_dict['numb_epoch'] # 10 - number of epoch
        self.batch_size = info_dict['batch_size']
        self.save_dir = info_dict['save_dir']
        self.save_path = info_dict.get('checkpoint', '')
        self.optimizer_choice = info_dict['optimizer']
        self.learning_rate = info_dict['learning_rate']
        self.grad_clip = info_dict['grad_clip']
        self.model_name = info_dict['model_name']
        self.checkpoint = info_dict['checkpoint']
        self.visual_backbone = info_dict['visual_backbone']
        self.include_pred_ft = info_dict['include_pred_ft']
        self.freeze = info_dict['freeze']
        self.tokenizer = AutoTokenizer.from_pretrained(info_dict['NLI_id'])
        self.obj_ft_dir = info_dict['obj_ft_dir']
        self.pred_ft_dir = info_dict['pred_ft_dir']
        self.OOC_type = info_dict['OOC_type']
        self.data_train_path = f"{DATA_DIR}/train/{self.OOC_type}"
        images_data_train = joblib.load(f"{self.data_train_path}/cheapfake_lowered_images_data_{info_dict['visual_id']}.joblib")
        caps_data_train = joblib.load(f"{self.data_train_path}/cheapfake_train_lowered_caps_data.joblib")
        df_train = pd.read_csv(f"{self.data_train_path}/train_{self.OOC_type}_quick.csv")
        self.datatrain = PairGraphPrecomputeDataset(image_sgg=images_data_train, caption_sgg=caps_data_train, 
                                                    word2idx_cap=word2idx_cap, word2idx_img_obj=word2idx_img_obj, word2idx_img_pred=word2idx_img_pred, 
                                                    effnet=self.visual_backbone, samples=df_train, tokenizer=self.tokenizer,
                                                    obj_ft_dir=self.obj_ft_dir, pred_ft_dir=self.pred_ft_dir)
        
        ## DECLARE MODEL
        self.model = md.CheapFake_Detection(info_dict=self.info_dict)
        
        self.model = self.model.to(device)
        
        if self.freeze: # freeze most of component
            for p in self.model.feature_extraction.parameters():
                p.requires_grad = False
            last_layer = self.model.MLP.numb_layers - 1
            for name, param in self.model.MLP.named_parameters():
                if name not in [f'linear.{last_layer}.weight', f'linear.{last_layer}.bias']:
                    param.requires_grad = False
                
        ## PARAMS & OPTIMIZER
        self.params = []
        self.params += list(filter(lambda p: p.requires_grad, self.model.parameters()))

        if self.optimizer_choice.lower() == 'adam':                                                     
            self.optimizer = optim.Adam(self.params,
                                         lr=self.learning_rate,
                                         betas=(0.9, 0.999),
                                         eps=1e-08,
                                         weight_decay=0)
                                      
        if self.optimizer_choice.lower() == 'sgd':
            self.optimizer = optim.SGD(self.params,
                                        lr=self.learning_rate,
                                        momentum=0.9,
                                        weight_decay=0)
            
    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR
           decayed by 10 every 30 epochs"""
        lr = self.learning_rate * (0.1 ** (epoch // 15)) # 15 epoch update once
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        
    # ---------- WRITE INFO TO TXT FILE ---------
    def extract_info(self):
        try:
            timestampLaunch = self.timestampLaunch
        except:
            timestampLaunch = 'undefined'
        model_info_log = open(f"{self.save_dir}/{self.model_name}-{timestampLaunch}-INFO.log", "w")
        result = f"===== {self.model_name} =====\n"
        result += print_dict(self.info_dict)
        model_info_log.write(result)
        model_info_log.close()
        
    # ---------- LOAD TRAINED MODEL ---------
    def load_trained_model(self):
        #---- Load checkpoint 
        if self.checkpoint is not None:
            print(f"LOAD PRETRAINED MODEL AT {self.checkpoint}")
            modelCheckpoint = torch.load(self.checkpoint)
            self.model.load_state_dict(modelCheckpoint['model_state_dict'])
            if not self.freeze:
                self.optimizer.load_state_dict(modelCheckpoint['optimizer_state_dict'])
        else:
            print("TRAIN FROM SCRATCH")
            
            
    # ---------- RUN TRAIN ---------
    def train(self):
        ## LOAD PRETRAINED MODEL ##
        scheduler = ReduceLROnPlateau(self.optimizer, factor = 0.2, patience=5, 
                                      mode = 'min', verbose=True, min_lr=1e-6)
        #scheduler_remaining_models = ReduceLROnPlateau(self.optimizer_remaining_models, factor = 0.5, patience=10, 
                                                  #mode = 'min', verbose=True, min_lr=1e-6)
        
        ## LOSS FUNCTION ##
        loss = BCELoss()
        loss = loss.to(device)

        ## REPORT ##
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        self.timestampLaunch = timestampDate + '-' + timestampTime
        # f_log = open(f"{self.save_dir}/{self.model_name}-{self.timestampLaunch}-REPORT.log", "w")
        writer = SummaryWriter(f'{self.save_dir}/{self.model_name}-{self.timestampLaunch}/')
        self.extract_info()
        
        ## TRAIN THE NETWORK ##
        #f1_max = 0
        lossMIN = 100000
        flag = 0
        count_change_loss = 0
        
        for epochID in range (self.numb_epoch):
            print(f"Training {epochID}/{self.numb_epoch-1}")

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
            
            # Update learning rate at each epoch
            #self.adjust_learning_rate(epochID)
            
            lossTrain = self.train_epoch(loss, writer, epochID)
            # lossVal = self.val_epoch(epochID, loss_matrix)
            with torch.no_grad():
                lossVal = self.validate_loss(images_data_val, caps_data_val, df_val, OBJ_FT_DIR_val, PRED_FT_DIR_val, loss)
                #if self.freeze:
                #else:
                    #lossVal = self.validate_retrieval(images_data_val, caps_data_val, include_geb=False)
                #lossTrain_recall = self.validate_retrieval(images_data_train, caps_data_train)
            #lossTrain_recall = 6 - lossTrain_recall
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            scheduler.step(lossVal)
            #scheduler_remaining_models.step(lossVal)
            info_txt = f"Epoch {epochID + 1}/{self.numb_epoch} [{timestampEND}]"
            
            if lossVal < lossMIN:# < lossMIN > f1_max
                count_change_loss = 0
                if lossVal < lossMIN:
                    #f1_max = lossVal
                    lossMIN = lossVal
                self.save_path = f"{self.save_dir}/{self.model_name}-{self.timestampLaunch}.pth.tar"
                torch.save({'epoch': epochID, \
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'best_loss': lossMIN}, 
                            self.save_path)#lossMIN f1_max
                info_txt = info_txt + f" [SAVE]\nLoss Val: {lossVal}"
               
            else:
                count_change_loss += 1
                info_txt = info_txt + f"\nLoss Val: {lossVal}"   
            print(info_txt)
            info_txt = info_txt + f"\nLoss Train: {round(lossTrain,6)}\n----------\n"
            with open(f"{self.save_dir}/{self.model_name}-{self.timestampLaunch}-REPORT.log", "a") as f_log:
                f_log.write(info_txt)
                    
            writer.add_scalars('Loss Epoch', {'train': lossTrain}, epochID)
            writer.add_scalars('Loss Epoch', {'train': lossTrain}, epochID)
            writer.add_scalars('Loss Epoch', {'val': lossVal}, epochID)
            writer.add_scalars('Loss Epoch', {'val-best': lossMIN}, epochID)#lossMIN
            
            current_lr = self.optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning Rate', current_lr, epochID)

            if count_change_loss >= 10:
                print(f'Early stopping: {count_change_loss} epoch not decrease the loss')
                break

        # f_log.close()
        writer.close()
        return self.save_path
    
    # ---------- TRAINING 1 EPOCH ---------
    def train_epoch(self, loss, writer, epochID):
        '''
        numb_sample = len(self.datatrain)
        temp = [x['label'] for x in self.datatrain]
        numb_match = np.sum(np.asarray(temp))
        numb_unmatch = numb_sample - numb_match
        print(f"Total Training sample: {numb_sample} --- Matched sample: {numb_match} --- UnMatched sample: {numb_unmatch}")
        '''
        
        dataloadertrain = make_PairGraphPrecomputeDataLoader(self.datatrain, batch_size=self.batch_size, num_workers=0)
        
        self.model.train()
        
        loss_report = 0
        count = 0
        numb_iter = len(dataloadertrain)
        print(f"Total iteration: {numb_iter}")
        for batchID, batch in enumerate(dataloadertrain):
            ## CUDA ##
            #if batchID > 2:
            #    break
            img_p_o, img_p_o_ft, img_p_p, img_p_p_ft, img_p_e, img_p_numb_o, img_p_numb_p,\
            cap_p_o_1, cap_p_p_1, cap_p_e_1, cap_p_numb_o_1, cap_p_numb_p_1, cap_p_len_p_1,\
            cap_p_o_2, cap_p_p_2, cap_p_e_2, cap_p_numb_o_2, cap_p_numb_p_2, cap_p_len_p_2,\
            cap_p_s, cap_p_m, cap_p_len_s, labels = batch
            
            batch_size = len(cap_p_len_s)
            
            img_p_o_ft = img_p_o_ft.to(device)
            img_p_p_ft = img_p_p_ft.to(device)
            img_p_o = img_p_o.to(device)
            img_p_p = img_p_p.to(device) 
            img_p_e = img_p_e.to(device)
            # cap_p_o = cap_p_o.to(device)
            # cap_p_p = cap_p_p.to(device) 
            cap_p_e_1 = cap_p_e_1.to(device)
            cap_p_e_2 = cap_p_e_2.to(device)
            labels = labels.to(device)
            #match_labels = match_labels.to(device)
            
            if not self.include_pred_ft:
                img_p_p_ft = None
                
            ## Calculate loss function
            #Binary cross entropy loss
            predict_labels = self.model(img_p_o_ft, img_p_p_ft, img_p_o, img_p_p, img_p_e, img_p_numb_o, img_p_numb_p, cap_p_s, cap_p_m, cap_p_p_1, cap_p_e_1, cap_p_len_p_1, cap_p_numb_o_1, cap_p_numb_p_1, cap_p_p_2, cap_p_e_2, cap_p_len_p_2, cap_p_numb_o_2, cap_p_numb_p_2)
            lossvalue_BCE = loss(predict_labels, labels)

            #cosine_loss = cos_loss(image_geb, caption_geb_2, match_labels)
            #print(batchID)
                
            ## LOSS
            # Assign to right format
            # img_obj [batch_size, max obj, dim], img_pred [batch_size, max pred, dim]
   
            #lossvalue = loss(predict_labels, labels)
            lossvalue = lossvalue_BCE

            #print(lossvalue)
            
            ## Update ##
            self.optimizer.zero_grad()
            lossvalue.backward()
            
            if self.grad_clip > 0:
                clip_grad_norm(self.params,
                               self.grad_clip)
            self.optimizer.step()
            loss_report += lossvalue.item()
            count += 1
            if (batchID+1) % 300 == 0:
                print(f"Batch Idx: {batchID+1} / {len(dataloadertrain)}: Loss Train {round(loss_report/count, 6)}")
                writer.add_scalars('Loss Training Iter', {'loss': loss_report/count}, epochID * np.floor(numb_iter/20) + np.floor((batchID+1)/20))
                
        return loss_report/count
    
    # ---------- VALIDATE ---------
    def validate_loss(self, image_sgg, caption_sgg, df, obj_ft_dir=OBJ_FT_DIR, pred_ft_dir=PRED_FT_DIR, loss=BCELoss()):
        print('---------- VALIDATE RETRIEVAL ----------')
        data_val = PairGraphPrecomputeDataset(image_sgg=image_sgg, caption_sgg=caption_sgg, 
                                                    word2idx_cap=word2idx_cap, word2idx_img_obj=word2idx_img_obj, word2idx_img_pred=word2idx_img_pred, 
                                                    obj_ft_dir=obj_ft_dir, pred_ft_dir=pred_ft_dir,
                                                    effnet=self.visual_backbone, samples=df, tokenizer=self.tokenizer)
        dataloaderval = make_PairGraphPrecomputeDataLoader(data_val, batch_size=self.batch_size, num_workers=0)
        self.model.eval()
        total_loss = 0
        predicted_labels = np.array([])
        true_labels = np.array([])
        with torch.no_grad():
            print('Embedding objects and predicates of images ...')
            for batchID, batch in enumerate(dataloaderval):
                img_p_o, img_p_o_ft, img_p_p, img_p_p_ft, img_p_e, img_p_numb_o, img_p_numb_p,\
                cap_p_o_1, cap_p_p_1, cap_p_e_1, cap_p_numb_o_1, cap_p_numb_p_1, cap_p_len_p_1,\
                cap_p_o_2, cap_p_p_2, cap_p_e_2, cap_p_numb_o_2, cap_p_numb_p_2, cap_p_len_p_2,\
                cap_p_s, cap_p_m, cap_p_len_s, labels = batch
                batch_size = len(cap_p_len_s)
                img_p_o_ft = img_p_o_ft.to(device)
                img_p_p_ft = img_p_p_ft.to(device)
                img_p_o = img_p_o.to(device)
                img_p_p = img_p_p.to(device) 
                img_p_e = img_p_e.to(device)
                # cap_p_o = cap_p_o.to(device)
                # cap_p_p = cap_p_p.to(device) 
                cap_p_e_1 = cap_p_e_1.to(device)
                cap_p_e_2 = cap_p_e_2.to(device)
                if not self.include_pred_ft:
                    img_p_p_ft = None
                ## Calculate loss function
                predict_labels = self.model(img_p_o_ft, img_p_p_ft, img_p_o, img_p_p, img_p_e, img_p_numb_o, img_p_numb_p, cap_p_s, cap_p_m, cap_p_p_1, cap_p_e_1, cap_p_len_p_1, cap_p_numb_o_1, cap_p_numb_p_1, cap_p_p_2, cap_p_e_2, cap_p_len_p_2, cap_p_numb_o_2, cap_p_numb_p_2) 
                predicted_labels = np.concatenate((predicted_labels, predict_labels.cpu().numpy().flatten()))#.item
                true_labels = np.concatenate((true_labels, labels.cpu().numpy().flatten()))#.item
                lossvalue = loss(predict_labels.to(device), labels.to(device))
                total_loss += lossvalue.item()
        #result = calculate_metric(true_labels, predicted_labels)
        return total_loss

# ----- EVALUATOR -----
class Evaluator():
    def __init__(self, info_dict):
        super(Evaluator, self).__init__()
        ##### INIT #####
        self.info_dict = info_dict
        self.info_dict['total_img_obj'] = TOTAL_IMG_OBJ
        self.info_dict['total_img_pred'] = TOTAL_IMG_PRED
        self.info_dict['total_cap_words'] = TOTAL_CAP_WORDS
        self.numb_sample = info_dict['numb_sample'] # 50000 - number of training sample in 1 epoch
        self.numb_epoch = info_dict['numb_epoch'] # 10 - number of epoch
        self.batch_size = info_dict['batch_size']
        self.save_dir = info_dict['save_dir']
        self.optimizer_choice = info_dict['optimizer']
        self.learning_rate = info_dict['learning_rate']
        self.grad_clip = info_dict['grad_clip']
        self.model_name = info_dict['model_name']
        self.checkpoint = info_dict['checkpoint']
        self.visual_backbone = info_dict['visual_backbone']
        self.include_pred_ft = info_dict['include_pred_ft']
        self.freeze = info_dict['freeze']
        self.tokenizer = AutoTokenizer.from_pretrained(info_dict['NLI_id'])
        
        ## DECLARE MODEL
        self.model = md.CheapFake_Detection(info_dict=self.info_dict)
        self.model = self.model.to(device)
        self.model.eval()
        
    # ---------- LOAD TRAINED MODEL ---------
    def load_trained_model(self):
        #---- Load checkpoint 
        if self.checkpoint is not None:
            print(f"LOAD PRETRAINED MODEL AT {self.checkpoint}")
            modelCheckpoint = torch.load(self.checkpoint)
            self.model.load_state_dict(modelCheckpoint['model_state_dict'], strict=False)
        else:
            print("TRAIN FROM SCRATCH")            
    
   # ---------- VALIDATE ---------
    def validate_loss(self, image_sgg=images_data_val, caption_sgg=caps_data_val, df_val=df_val, obj_ft_dir=OBJ_FT_DIR, pred_ft_dir=PRED_FT_DIR, loss=BCELoss()):
        print('---------- VALIDATE RESULT ----------')
        data_val = PairGraphPrecomputeDataset(image_sgg=image_sgg, caption_sgg=caption_sgg, 
                                                    word2idx_cap=word2idx_cap, word2idx_img_obj=word2idx_img_obj, word2idx_img_pred=word2idx_img_pred, 
                                                    effnet=self.visual_backbone, samples=df_val, tokenizer=self.tokenizer,
                                                    obj_ft_dir=obj_ft_dir, pred_ft_dir=pred_ft_dir)
        dataloaderval = make_PairGraphPrecomputeDataLoader(data_val, batch_size=1, num_workers=0)
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            print('Embedding objects and predicates of images ...')
            for batchID, batch in enumerate(dataloaderval):
                img_p_o, img_p_o_ft, img_p_p, img_p_p_ft, img_p_e, img_p_numb_o, img_p_numb_p,\
                cap_p_o_1, cap_p_p_1, cap_p_e_1, cap_p_numb_o_1, cap_p_numb_p_1, cap_p_len_p_1,\
                cap_p_o_2, cap_p_p_2, cap_p_e_2, cap_p_numb_o_2, cap_p_numb_p_2, cap_p_len_p_2,\
                cap_p_s, cap_p_m, cap_p_len_s, labels = batch
                batch_size = len(cap_p_len_s)
                img_p_o_ft = img_p_o_ft.to(device)
                img_p_p_ft = img_p_p_ft.to(device)
                img_p_o = img_p_o.to(device)
                img_p_p = img_p_p.to(device) 
                img_p_e = img_p_e.to(device)
                # cap_p_o = cap_p_o.to(device)
                # cap_p_p = cap_p_p.to(device) 
                cap_p_e_1 = cap_p_e_1.to(device)
                cap_p_e_2 = cap_p_e_2.to(device)
                if not self.include_pred_ft:
                    img_p_p_ft = None
                ## Calculate loss function
                predict_labels = self.model(img_p_o_ft, img_p_p_ft, img_p_o, img_p_p, img_p_e, img_p_numb_o, img_p_numb_p, cap_p_s, cap_p_m, cap_p_p_1, cap_p_e_1, cap_p_len_p_1, cap_p_numb_o_1, cap_p_numb_p_1, cap_p_p_2, cap_p_e_2, cap_p_len_p_2, cap_p_numb_o_2, cap_p_numb_p_2) 
                lossvalue = loss(predict_labels.to(device), labels.to(device))
                total_loss += lossvalue.item()
        return total_loss
    
    # ---------- VALIDATE RESULT ---------
    def validate_result(self, image_sgg=images_data_val, caption_sgg=caps_data_val, df_val=df_val, obj_ft_dir=OBJ_FT_DIR, pred_ft_dir=PRED_FT_DIR):
        print('---------- CALCULATE F1-SCORE ----------')
        data_val = PairGraphPrecomputeDataset(image_sgg=image_sgg, caption_sgg=caption_sgg, 
                                                    word2idx_cap=word2idx_cap, word2idx_img_obj=word2idx_img_obj, word2idx_img_pred=word2idx_img_pred, 
                                                    effnet=self.visual_backbone, samples=df_val, 
                                                    obj_ft_dir=obj_ft_dir, pred_ft_dir=pred_ft_dir, tokenizer=self.tokenizer)
        dataloaderval = make_PairGraphPrecomputeDataLoader(data_val, batch_size=1, num_workers=0, shuffle=False)
        self.model.eval()
        predicted_labels = np.array([])
        true_labels = np.array([])
        with torch.no_grad():
            print('Embedding objects and predicates of images ...')
            for batchID, batch in enumerate(dataloaderval):
                img_p_o, img_p_o_ft, img_p_p, img_p_p_ft, img_p_e, img_p_numb_o, img_p_numb_p,\
                cap_p_o_1, cap_p_p_1, cap_p_e_1, cap_p_numb_o_1, cap_p_numb_p_1, cap_p_len_p_1,\
                cap_p_o_2, cap_p_p_2, cap_p_e_2, cap_p_numb_o_2, cap_p_numb_p_2, cap_p_len_p_2,\
                cap_p_s, cap_p_m, cap_p_len_s, label = batch
                batch_size = len(cap_p_len_s)
                img_p_o_ft = img_p_o_ft.to(device)
                img_p_p_ft = img_p_p_ft.to(device)
                img_p_o = img_p_o.to(device)
                img_p_p = img_p_p.to(device) 
                img_p_e = img_p_e.to(device)
                # cap_p_o = cap_p_o.to(device)
                # cap_p_p = cap_p_p.to(device) 
                cap_p_e_1 = cap_p_e_1.to(device)
                cap_p_e_2 = cap_p_e_2.to(device)
                if not self.include_pred_ft:
                    img_p_p_ft = None
                ## Calculate loss function
                predict_labels = self.model(img_p_o_ft, img_p_p_ft, img_p_o, img_p_p, img_p_e, img_p_numb_o, img_p_numb_p, cap_p_s, cap_p_m, cap_p_p_1, cap_p_e_1, cap_p_len_p_1, cap_p_numb_o_1, cap_p_numb_p_1, cap_p_p_2, cap_p_e_2, cap_p_len_p_2, cap_p_numb_o_2, cap_p_numb_p_2)
                predicted_labels = np.append(predicted_labels, predict_labels.item())
                true_labels = np.append(true_labels, label.item())
        result_df = pd.DataFrame({"predict_percent":predicted_labels, "predict": (predicted_labels>=0.5).astype(int), "true": true_labels})
        date = time.strftime("%d%m%Y")
        result_df.to_csv(f"Predict_vs_true_{date}.csv")
        result = calculate_metric(true_labels, predicted_labels)
        return result
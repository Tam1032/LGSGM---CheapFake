from Controller_GCN import Trainer, Evaluator
import torch
import os
import joblib
import pandas as pd
import wandb

wandb.login()
DATA_DIR = "./Data"
info_dict = dict()
info_dict['save_dir'] = './Report'
info_dict['numb_sample'] = None # training sample for 1 epoch
info_dict['numb_epoch'] = 50 #100 # number of epoch
info_dict['numb_gcn_layers'] = 1 # number of gin layers to be stacked
info_dict['gcn_hidden_dim'] = [] # hidden layer in each gin layer
info_dict['gcn_output_dim'] = 1024 # graph embedding final dim
info_dict['gcn_input_dim'] = 2048 # node and edges dim of a graph
info_dict['batchnorm'] = True
info_dict['batch_size'] = 32
info_dict['dropout'] = 0.5
info_dict['visual_backbone'] = 'b5' # EfficientNet backbone to extract visual features
info_dict['visual_ft_dim'] = 2048
info_dict['optimizer'] = 'Adam' # or Adam
info_dict['learning_rate'] = 3e-5
info_dict['activate_fn'] = 'swish' # swish, relu, leakyrelu
info_dict['grad_clip'] = 2 # Gradient clipping
# info_dict['use_residual'] = False # always set it to false (not implemented yet)
# Embedder for each objects and predicates, embed graph only base on objects
info_dict['model_name'] = 'GCN_ObjAndPredShare_NoFtExModule_LSTM' 
info_dict['checkpoint'] = None # Training from a pretrained path
info_dict['margin_matrix_loss'] = 0.35
info_dict['rnn_numb_layers'] = 2
info_dict['rnn_bidirectional'] = True
info_dict['rnn_structure'] = 'LSTM' # LSTM or GRU (LSTM gave better result)
info_dict['graph_emb_dim'] = info_dict['gcn_output_dim']*2
info_dict['include_pred_ft'] = True # include visual predicate features or not
info_dict['freeze'] = True #False # Freeze all layers except the graph convolutional network and graph embedding module

def run_train(info_dict):
    #info_dict['checkpoint'] = './Report/GCN_ObjAndPredShare_NoFtExModule_LSTM-train_10000.pth.tar'
    if not os.path.exists(info_dict['save_dir']):
        print(f"Creating {info_dict['save_dir']} folder")
        os.makedirs(info_dict['save_dir'])
        
    trainer = Trainer(info_dict)
    checkpoint_path = trainer.train()

    subset = 'test'
    DATA_DIR = './Data'
    visual_type = info_dict['visual_id']
    images_data = joblib.load(f"{DATA_DIR}/{subset}/cheapfake_{subset}_lowered_images_data_{visual_type}.joblib")
    caps_data = joblib.load(f"{DATA_DIR}/{subset}/cheapfake_{subset}_lowered_caps_data.joblib")
    df = pd.read_csv(f"{DATA_DIR}/{subset}/label_file_{subset}.csv")
    obj_ft_dir_val = f"./Data/test/{visual_type}/VisualObjectFeatures"
    pred_ft_dir_val = f'./Data/test/{visual_type}/VisualPredFeatures'
    
    lossVal = trainer.validate_loss(images_data, caps_data, df, obj_ft_dir_val, pred_ft_dir_val)
    info_txt = f"Loss Val: {lossVal}" 
    print(info_txt)
    return lossVal, checkpoint_path
    
def run_evaluate(info_dict, checkpoint_path):
    # path to pretrained model
    info_dict['checkpoint'] = checkpoint_path

    evaluator = Evaluator(info_dict)
    evaluator.load_trained_model()
    
    subset = 'test'
    visual_type = info_dict['visual_id']
    DATA_DIR = './Data'
    images_data = joblib.load(f"{DATA_DIR}/{subset}/cheapfake_{subset}_lowered_images_data_{visual_type}.joblib")
    caps_data = joblib.load(f"{DATA_DIR}/{subset}/cheapfake_{subset}_lowered_caps_data.joblib")
    df = pd.read_csv(f"{DATA_DIR}/{subset}/label_file_{subset}.csv")
    OBJ_FT_DIR = f"{DATA_DIR}/{subset}/{visual_type}/VisualObjectFeatures" # run extract_visual_features.py to get this
    PRED_FT_DIR = f"{DATA_DIR}/{subset}/{visual_type}/VisualPredFeatures" # run extract_visual_features.py to get this
    
    #lossVal, ar_val, ari_val = evaluator.validate_retrieval(images_data, caps_data, False)
    #info_txt = f"\n----- SUMMARY (Matrix)-----\nLoss Val: {6-lossVal}"   
    #info_txt = info_txt + f"\n[i2t] {round(ar_val[0], 4)} {round(ar_val[1], 4)} {round(ar_val[2], 4)}"
    #info_txt = info_txt + f"\n[t2i] {round(ari_val[0], 4)} {round(ari_val[1], 4)} {round(ari_val[2], 4)}"
    #print(info_txt)
    
    lossVal = evaluator.validate_loss(images_data, caps_data, df, obj_ft_dir=OBJ_FT_DIR, pred_ft_dir=PRED_FT_DIR)
    result = evaluator.validate_result(images_data, caps_data, df, obj_ft_dir=OBJ_FT_DIR, pred_ft_dir=PRED_FT_DIR)
    info_txt = f"\n----- SUMMARY (Combine)-----\nLoss Val: {lossVal}"   
    info_txt = info_txt + f"\nF1_score: {round(result['f1'], 4)}"
    info_txt = info_txt + f"\nPrecision: {round(result['precision'], 4)}"
    info_txt = info_txt + f"\nRecall: {round(result['recall'], 4)}"
    info_txt = info_txt + f"\nAccuracy: {round(result['accuracy'], 4)}"
    print(info_txt)
    return result

OOC_types = ["type1", "type2", "type3"]
visual_ids = ["Pe-NET", "Neural-Motifs"]
textual_ids = ["MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "facebook/bart-large-mnli", "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"]
for visual_id in visual_ids:
    for textual_id in textual_ids:
        for OOC_type in OOC_types:
            info_dict['NLI_id'] = textual_id
            info_dict['visual_id'] = visual_id
            info_dict['OOC_type'] = OOC_type
            subset = "train"
            text_model = textual_id.split("/")[1].split("-")[0]
            info_dict['obj_ft_dir'] = f'{DATA_DIR}/{subset}/{visual_id}/VisualObjectFeatures'
            info_dict['pred_ft_dir'] = f'{DATA_DIR}/{subset}/{visual_id}/VisualPredFeatures'
            # Init wandb tracking
            wandb.init(
                project="Cheapfake_Revision_1",
                name=f"{visual_id}_{text_model}_{OOC_type}",
                config=info_dict
            )
            # Running experiments
            loss_val, checkpoint_path = run_train(info_dict)
            torch.cuda.empty_cache()
            #checkpoint_path = "Report/GCN_ObjAndPredShare_NoFtExModule_LSTM-10022025-213619.pth.tar"
            result = run_evaluate(info_dict, checkpoint_path)
            wandb.log({"loss validation": loss_val})
            wandb.log(result)
            # Mark the run as finished
            wandb.finish()
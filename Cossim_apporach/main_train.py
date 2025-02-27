from Controller_GCN import Trainer, Evaluator
import torch
import os
import joblib
import pandas as pd

info_dict = dict()
info_dict['save_dir'] = './Report'


info_dict['numb_sample'] = None # training sample for 1 epoch
info_dict['numb_epoch'] = 25 # number of epoch
info_dict['numb_gcn_layers'] = 1 # number of gin layers to be stacked
info_dict['gcn_hidden_dim'] = [] # hidden layer in each gin layer
info_dict['gcn_output_dim'] = 1024 # graph embedding final dim
info_dict['gcn_input_dim'] = 2048 # node and edges dim of a graph
info_dict['batchnorm'] = True
info_dict['batch_size'] = 16
info_dict['dropout'] = 0.5
info_dict['visual_backbone'] = 'b5' # EfficientNet backbone to extract visual features
info_dict['visual_ft_dim'] = 2048
info_dict['optimizer'] = 'Adam' # or Adam
info_dict['learning_rate'] = 3e-4 #3e-4
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
info_dict['freeze'] = True # Freeze all layers except the graph convolutional network and graph embedding module

def run_train(info_dict):
    info_dict['encode_weights'] = './Report/GCN_CheapFake_weights_Bart_12000_NeuralMotif.pth.tar'
    if not os.path.exists(info_dict['save_dir']):
        print(f"Creating {info_dict['save_dir']} folder")
        os.makedirs(info_dict['save_dir'])
        
    trainer = Trainer(info_dict)
    trainer.train()

    subset = 'test'
    DATA_DIR = './Data'
    images_data = joblib.load(f"{DATA_DIR}/{subset}/cheapfake_{subset}_lowered_images_data_neural_motif.joblib")#_neural_motif
    caps_data = joblib.load(f"{DATA_DIR}/{subset}/cheapfake_{subset}_lowered_caps_data.joblib")
    df = pd.read_csv(f"{DATA_DIR}/{subset}/label_file_{subset}.csv")
    obj_ft_dir_val = "./Data/test/Neural_Motif/VisualObjectFeatures"#Neural_Motif PENET
    pred_ft_dir_val = './Data/test/Neural_Motif/VisualPredFeatures'#Neural_Motif
    
    lossVal = trainer.validate_loss(images_data, caps_data, df, obj_ft_dir_val, pred_ft_dir_val)
    info_txt = f"Loss Val: {lossVal}" 
    print(info_txt)
    
def run_evaluate(info_dict):
    # path to pretrained model
    info_dict['checkpoint'] = ''

    evaluator = Evaluator(info_dict)
    evaluator.load_trained_model()
    
    subset = 'test'
    DATA_DIR = './Data'
    #images_data = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_images_data.joblib")
    #caps_data = joblib.load(f"{DATA_DIR}/flickr30k_{subset}_lowered_caps_data.joblib")
    #images_data = joblib.load(f"{DATA_DIR}/visual_test_100_images.joblib")
    #caps_data = joblib.load(f"{DATA_DIR}/caption_test_100_images.joblib")
    images_data = joblib.load(f"{DATA_DIR}/{subset}/cheapfake_{subset}_lowered_images_data_neural_motif.joblib")#_neural_motif
    caps_data = joblib.load(f"{DATA_DIR}/{subset}/cheapfake_{subset}_lowered_caps_data.joblib")
    df = pd.read_csv(f"{DATA_DIR}/{subset}/label_file_{subset}.csv")
    OBJ_FT_DIR = f"{DATA_DIR}/{subset}/Neural_Motif/VisualObjectFeatures" # run extract_visual_features.py to get this
    PRED_FT_DIR = f"{DATA_DIR}/{subset}/Neural_Motif/VisualPredFeatures" # run extract_visual_features.py to get this
    
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
    
#run_train(info_dict)
run_evaluate(info_dict)
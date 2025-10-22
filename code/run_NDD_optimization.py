'''
Goal here is to run hyperparamter grid searches for the parameters i've identified for each model.
Fixed
Num channels
Changes per clip
How many clips do I actually need to do this for? Maybe just do the R2 tests on like 5 clips with variable number of channels to see if long-term forecasting/deeper models improve with more training data
Patience
Set to 3
Learning rate
0.0005
Validation percentage
10%
Sampling rate
256
16 samples is 63 ms (62.5)
Seeds
1:3
Max epochs
200
All
Sequence length
1,2,4,8,16,32,64
Forecast length
1,2,4,8,16,32,64
Training set size
5, 10, 30, 60, 120, 180, 300, 600
LiNDDA
None
Deeper models
Num layers
1, 2, 4, 8
Num parameters
Num_channels * seq_length / num_layers
1 (this is "constrained" to the same number of parameters as the linear model)
2
4
8

'''
import os
from os.path import join as ospj
from pqdm.processes import pqdm
from tqdm import tqdm
import sys
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import time
import pickle
# Get the project root (parent directory of examples/)
script_dir = os.path.dirname(os.path.abspath(__file__))
dynasd_root = os.path.join(script_dir, '..', '..', 'DynaSD')

if dynasd_root not in sys.path:
    sys.path.insert(0, dynasd_root)

import pandas as pd
from utils import get_data_from_bids, preprocess_for_detection, remove_scalp_electrodes, clean_labels

from config import Config
datapath,prodatapath = Config.deal(['datapath','prodatapath'])

from DynaSD import GIN, LiNDDA, MINDD, LiRNDDA
verbose = False
def evaluate_model(param_dict):
    sequence_length = param_dict['sequence_length']
    forecast_length = 1
    
    # Check if checkpoint already exists
    os.makedirs(ospj(prodatapath,"ndd_checkpoints"),exist_ok=True)
    
    mdl_str = param_dict['model'].__name__
    num_layers = param_dict['num_layers'] if param_dict['num_layers'] is not None else 'None'
    param_scale = param_dict['param_scale'] if param_dict['param_scale'] is not None else 'None'
    hidden_size = param_dict['hidden_size'] if param_dict['hidden_size'] is not None else 'None'
    num_stacks = param_dict['num_stacks'] if param_dict['num_stacks'] is not None else 'None'
    duration = param_dict['duration']
    
    filename = f"{param_dict['patient']}_{duration}_{mdl_str}_{param_dict['sequence_length']}_{num_layers}_{param_scale}_{hidden_size}_{num_stacks}.pkl"
    checkpoint_path = ospj(prodatapath,"ndd_checkpoints", filename)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading existing checkpoint: {filename}")
        return pd.read_pickle(checkpoint_path)
    
    train_time = time.perf_counter()
    if param_dict['model'] == LiNDDA:
        reg = LiNDDA(
            sequence_length = sequence_length,
            forecast_length = forecast_length,
            patience = 2,
            lr = 0.0005,
            val_split = 0.1,
            fs = 256,
            early_stopping = True,
            num_epochs = 500,
            batch_size = batch_size,
            verbose = verbose,
            use_cuda = False,
        )
    elif param_dict['model'] == MINDD:
        reg = param_dict['model'](
            fs = 256,
            sequence_length = sequence_length,
            forecast_length = forecast_length,
            hidden_sizes = [param_dict['hidden_size']]*param_dict['num_layers'],
            patience = 2,
            lr = 0.0005,
            val_split = 0.1,
            early_stopping = True,
            num_epochs = 200,
            batch_size = batch_size,
            verbose = verbose,
            use_cuda = False,
        )
    else:
        reg = param_dict['model'](
            fs = 256,
            sequence_length = sequence_length,
            forecast_length = forecast_length,
            # hidden_size = param_dict['data'].shape[1]*2,
            hidden_size = param_dict['hidden_size'],
            num_layers = param_dict['num_layers'],
            num_stacks = param_dict['num_stacks'],
            num_epochs = 200,
            batch_size = batch_size,
            patience = 2,
            lr = 0.001,
            val_split = 0.1,
            early_stopping = True,
            verbose = verbose,
            use_cuda = False,
        )
    val_idx = int(param_dict['data'].shape[0] * 0.1)
    results_dicts = []
    reg.fit(param_dict['data'])
    train_time = time.perf_counter() - train_time
    bs = reg.batch_size
    num_epochs = reg.early_stop_epoch
    mdl_str = param_dict.pop('model').__name__

    for forecast_length in [1]:
        reg.forecast_length = forecast_length
        latency_s = time.perf_counter()
        x_pred = reg.predict(param_dict['data'])
        x_pred = x_pred[sequence_length:,:]

        train_r2 = r2_score(param_dict['data'].iloc[sequence_length:val_idx,:], x_pred[:val_idx-sequence_length,:])
        train_loss = mean_squared_error(param_dict['data'].iloc[sequence_length:val_idx,:],x_pred[:val_idx-sequence_length,:])
        
        coverage_gap = (len(param_dict['data']) - sequence_length) % forecast_length
        if coverage_gap != 0:
            val_r2 = r2_score(param_dict['data'].iloc[val_idx:-coverage_gap,:], x_pred[val_idx-sequence_length:-coverage_gap,:])
            val_loss = mean_squared_error(param_dict['data'].iloc[val_idx:-coverage_gap,:],x_pred[val_idx-sequence_length:-coverage_gap,:])
        else:
            val_r2 = r2_score(param_dict['data'].iloc[val_idx:,:], x_pred[val_idx-sequence_length:,:])
            val_loss = mean_squared_error(param_dict['data'].iloc[val_idx:,:],x_pred[val_idx-sequence_length:,:])
        
        x_test = reg.predict(param_dict['test'])
        x_test = x_test[sequence_length:,:]
        coverage_gap = (len(param_dict['test']) - sequence_length) % forecast_length
        if coverage_gap != 0:
            test_r2 = r2_score(param_dict['test'].iloc[sequence_length:-coverage_gap,:], x_test[:-coverage_gap,:])
            test_loss = mean_squared_error(param_dict['test'].iloc[sequence_length:-coverage_gap,:],x_test[:-coverage_gap,:])
        else:
            test_r2 = r2_score(param_dict['test'].iloc[sequence_length:,:], x_test[:,:])
            test_loss = mean_squared_error(param_dict['test'].iloc[sequence_length:,:],x_test[:,:])
        del x_pred
        latency_s = time.perf_counter() - latency_s
        results_dicts.append(dict(model = mdl_str, forecast_length = forecast_length, train_r2 = train_r2, 
        val_r2 = val_r2, test_r2 = test_r2, train_loss = train_loss, val_loss = val_loss, test_loss = test_loss, 
        batch_size = bs, num_epochs = num_epochs, latency_s = latency_s, train_s = train_time))

    param_dict.pop('data')
    param_dict.pop('test')
    results = [{**param_dict, **rd} for rd in results_dicts]
    del reg
    ret = pd.DataFrame(results)

    # Handle None values in filename
    num_layers = param_dict['num_layers'] if param_dict['num_layers'] is not None else 'None'
    param_scale = param_dict['param_scale'] if param_dict['param_scale'] is not None else 'None'
    hidden_size = param_dict['hidden_size'] if param_dict['hidden_size'] is not None else 'None'
    num_stacks = param_dict['num_stacks'] if param_dict['num_stacks'] is not None else 'None'
    
    filename = f"{param_dict['patient']}_{duration}_{mdl_str}_{param_dict['sequence_length']}_{num_layers}_{param_scale}_{hidden_size}_{num_stacks}.pkl"
    ret.to_pickle(ospj(prodatapath,"ndd_checkpoints", filename))
    return ret
    # return pd.Series(param_dict | dict(model = mdl_str, train_r2 = train_r2, val_r2 = val_r2, test_r2 = test_r2, train_loss = train_loss, val_loss = val_loss, test_loss = test_loss, batch_size = bs, num_epochs = num_epochs, latency_s = latency_s))

param_dict_list = []
batch_size = 2048
for pt in ['HUP126','HUP221','HUP276']:
# for pt in ['HUP065','HUP078','HUP126','HUP221','HUP276']:
    # Load in the ieeg clip
    X,fs_raw = get_data_from_bids(ospj(datapath,"BIDS"),pt,'interictal')
    X.columns = clean_labels(X.columns,pt)
    neural_channels = remove_scalp_electrodes(X.columns)

    X,fs_raw,_ = preprocess_for_detection(X.loc[:,neural_channels],fs_raw)
    n_channels = X.shape[1]
    # SAMPLE DATA FOR TESTING
# for _ in range(1):
#     pt = 'test'
    # X,fs_raw = pd.DataFrame(np.random.randn(600*256,8)),256
    # n_channels = X.shape[1]
    # END SAMPLE DATA FOR TESTING

    for x_len in [30, 60, 90, 120]:# 300, 540]:
    # for x_len in [90]:
        # using the fs from the BIDS, clip the Xtrain and assign it to state dict
        for sequence_length in [1,2,8,16,32]: #,64,128]:
            for num_layers in [1,2]:
                
                # for param_scale in [1,2,3]:

                #     hidden_size = int(sequence_length * n_channels * param_scale // num_layers)
                #     param_dict_list.append(
                #         dict(
                #                 data = X.iloc[:x_len*fs_raw,:],
                #                 test = X.iloc[60*fs_raw:,:],
                #                 patient = pt,
                #                 n_channels = n_channels,
                #                 duration = x_len,
                #                 sequence_length = sequence_length,
                #                 num_layers = num_layers,
                #                 param_scale = param_scale,
                #                 hidden_size = hidden_size,
                #                 num_stacks = None,
                #                 model = MINDD
                #             )
                #     )

                for num_stacks in [1]:
                    for hidden_size in [10,int(n_channels)]:
                        for model in [GIN]: # LiRNDDA
                            param_dict_list.append(
                                dict(
                                    data = X.iloc[:x_len*fs_raw,:],
                                    test = X.iloc[60*fs_raw:,:],
                                    patient = pt,
                                    n_channels = n_channels,
                                    duration = x_len,
                                    sequence_length = sequence_length,
                                    num_layers = num_layers,
                                    param_scale = None,
                                    hidden_size = hidden_size,
                                    num_stacks = num_stacks,
                                    model = model
                                )
                            )
            # param_dict_list.append(
                # dict(
                #     data = X.iloc[:x_len*fs_raw,:],
                #     test = X.iloc[60*fs_raw:,:],
                #     patient = pt,
                #     n_channels = n_channels,
                #     duration = x_len,
                #     sequence_length = sequence_length,
                #     num_layers = None,
                #     param_scale = None,
                #     hidden_size = None,
                #     num_stacks = None,
                #     model = LiNDDA
                # )
            # )

# res_series_list = pqdm(param_dict_list, evaluate_model, n_jobs=6)

res_series_list = []
for param_dict in tqdm(param_dict_list):
    res_series_list.append(evaluate_model(param_dict))

# pickle.dump(res_series_list, open(ospj(prodatapath,'res_series_list_deep_2_checkpoint.pkl'), 'wb'))
res_series_list = pd.concat([r for r in res_series_list if not isinstance(r, Exception)])
res_series_list.to_csv(ospj(prodatapath,'res_series_list_deep_3.csv'))


# for param_dict in param_dict_list:
#     print(param_dict)
#     x = evaluate_model(param_dict)
#     print(x)
#     break

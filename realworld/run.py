import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import os
import numpy as np
import time

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='iTransformer')
    
    parser.add_argument('--is_norm', action='store_false')
    parser.add_argument('--is_ln', action='store_false')
    parser.add_argument('--patch_size', type=int, default=5)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--c_type', type=str, default='None')

    parser.add_argument('--mode', type=str, default='regre', help='different mode of trend prediction block: [regre or mean]')
    parser.add_argument('--conv_kernel', type=int, nargs='+', default=[17,49], help='downsampling and upsampling convolution kernel_size')

    parser.add_argument('--draw', type=int, default=1, help='draw')
    
    parser.add_argument('--direction', type=int, default=1)
    parser.add_argument('--disturbance', type=float, default=0.0)
    parser.add_argument('--dimension', type=int, default=0)

    parser.add_argument('--seed', type=int, default=2024, help='seed')
    
    parser.add_argument('--save_path', type=str, default='IDOL_results')
    parser.add_argument('--WITRAN_deal', type=str, default='standard',
                        help='WITRAN deal data type, options:[None, standard]')
    parser.add_argument('--WITRAN_grid_cols', type=int, default=25,
                        help='Numbers of data grid cols for WITRAN')
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    parser.add_argument('--num_nodes', type=int, default=21)
    parser.add_argument('--layer_nums', type=int, default=3)
    parser.add_argument('--k', type=int, default=2, help='choose the Top K patch size at the every layer ')
    parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4])
    parser.add_argument('--patch_size_list', nargs='+', type=int, default=[5, 25, 5, 25, 5, 25, 5, 25, 5, 25, 5, 25])
    parser.add_argument('--revin', type=int, default=1, help='whether to apply RevIN')
    parser.add_argument('--drop', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--residual_connection', type=int, default=0)
    parser.add_argument('--metric',type=str, default='mae')

    parser.add_argument('--fc_dropout', type=float, default=0.3, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.3, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--momentum', type=float, default=0.1, help='momentum')
    parser.add_argument('--local_rank', type=int,default = 0)
    parser.add_argument('--devices_number',type=int,default = 1)
    parser.add_argument('--use_statistic',action='store_true', default=False)
    parser.add_argument('--use_decomp',action='store_true', default=False)
    parser.add_argument('--same_smoothing',action='store_true', default=False)
    parser.add_argument('--warmup_epochs',type=int,default = 0)
    parser.add_argument('--weight_decay',type=float,default = 0)
    parser.add_argument('--merge_size',type=int,default = 2)
    parser.add_argument('--use_untoken',type=int,default = 0)
    parser.add_argument('--pct_start',type=float,default = 0.3)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--dp_rank', type=int,default = 8)
    parser.add_argument('--rescale', type=int,default = 1)

    parser.add_argument('--zc_dim', type=int, default=51, help='num of encoder layers')
    parser.add_argument('--zd_dim', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--zd_kl_weight', type=float, default=0.001, help='num of encoder layers')
    # parser.add_argument('--zc_kl_weight', type=float, default=0.001, help='num of encoder layers')
    parser.add_argument('--sparsity_weight', type=float, default=0.001, help='num of encoder layers')
    parser.add_argument('--recon_weight', type=float, default=0.001, help='num of encoder layers')
    parser.add_argument('--kld_normal_weight', type=float, default=1e-7, help='num of encoder layers')
    parser.add_argument('--kld_future_weight', type=float, default=1e-7, help='num of encoder layers')
    parser.add_argument('--kld_weight', type=float, default=1e-7, help='num of encoder layers')
    parser.add_argument('--hmm_weight', type=float, default=0.001, help='num of encoder layers')
    parser.add_argument('--rec_weight', type=float, default=0.5, help='latent dimension of koopman embedding')
    parser.add_argument('--n_class', type=int, default=4, help='num of encoder layers')
    parser.add_argument('--No_prior', action='store_true', default=False, help='num of encoder layers')
    parser.add_argument('--lags', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--embedding_dim', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--dynamic_dim', type=int, default=128, help='latent dimension of koopman embedding')
    parser.add_argument('--pre_epoches', type=int, default=1, help='train epochs')

    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    parser.add_argument('--train_mode', type=int, default=0)
    parser.add_argument('--cut_freq', type=int, default=0)
    parser.add_argument('--base_T', type=int, default=24)
    parser.add_argument('--H_order', type=int, default=2)

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='IDOL',
                        help='model name')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/human/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='Walking_all.npy', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=125, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=125, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=125, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # rec enc
    parser.add_argument('--is_bn', action='store_true', default=False, help='num of encoder layers')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of en/decoder')
    parser.add_argument('--hidden_layers', type=int, default=2, help='number of hidden layers of en/decoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')
    parser.add_argument('--d_state', type=int, default=32, help='parameter of Mamba Block')
    parser.add_argument('--z_dim', type=int, default=10)

    parser.add_argument('--kl_weight', type=float, default=0.0001, help='num of encoder layers')



    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.kld_normal_weight = args.kld_feature_weight = args.kld_weight
    
    """
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    """
    if args.cut_freq == 0:
        args.cut_freq = int(args.seq_len // args.base_T + 1) * args.H_order + 10

    print('Args in experiment:')
    print(args)
    os.system('CUDA_VISIBLE_DEVICES=0')

    Exp = Exp_Long_Term_Forecast
    
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy('file_system')

    if args.is_training==1:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}i_{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii, args.learning_rate, args.batch_size, args.seed)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    elif args.is_training == 2:
        print(11111)
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)
            exp = Exp(args)
            exp.get_input(setting)
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

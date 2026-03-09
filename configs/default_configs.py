import ml_collections
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()
    
    # training
    config.training = training = ml_collections.ConfigDict()

    # inference
    config.inference = inference = ml_collections.ConfigDict()
    inference.queue_length = 512
    inference.threshold_type = "adaptive" 
    if inference.threshold_type == "fixed":
        inference.fixed_threshold = 0.5
    inference.top = 1. # 0.5

    config.data = data = ml_collections.ConfigDict()
    data.path = "../csp_adaneg/data/images_largescale"
    data.OOD_ratio = 1.
    data.workers = 8

    # model
    config.model = model = ml_collections.ConfigDict()
    model.arch = "ViT-B/16" 
    model.resolution = 224 
    model.n_ctx = 4
    model.ctx_init = "a_photo_of_a"

    # optimization
    config.optim = optim = ml_collections.ConfigDict()

    # logs
    config.logs = logs = ml_collections.ConfigDict()
    logs.project = 'zeroshot_noisyTTDA'
    logs.path = 'results/'
    logs.visualization = 'data_analysis/score'
    logs.conf_path = 'data_analysis/conf_pkl'
    logs.grads_path = 'data_analysis/grads_pkl'
    
    logs.experiment_group = 'main_results'

    config.anlysis_mode = 'normal_mode' 
    config.seed = 0
    config.print_freq = 200
    config.device = (torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu"))

    return config
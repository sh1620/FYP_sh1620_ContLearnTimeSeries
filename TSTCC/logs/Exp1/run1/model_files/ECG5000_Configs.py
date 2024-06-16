class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 320 #set to 320 according to ts2vec paper about tstcc comparison

        self.num_classes = 5
        self.dropout = 0.35
        self.features_len = 20

        # training configs
        self.num_epoch = 40 #200 caused overfitting

        
        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 128 #batch size is 128

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

#TS2Vec paper said when they tested ECG5000 with TSTCC they used configurations from HAR dataset
class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 6


import os

ROOT_DATASET = '/home/kevinq/datasets/'
    

def return_kineticszsar(modality):
    n_class = 768
    if modality == 'RGB':
        root_data = "./datasets/kinetics_zsar"
        filename_imglist_train = './datasets/kinetics_zsar/trn.json'
        filename_imglist_val = './datasets/kinetics_zsar/val.json'
        filename_imglist_test = './datasets/kinetics_zsar/tst.json'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return n_class, filename_imglist_train, filename_imglist_val, filename_imglist_test, root_data

def return_keval(modality):
    n_class = 768
    if modality == 'RGB':
        root_data = "./datasets/kinetics_eval"
        filename_imglist_train = './datasets/kinetics_eval/tst1.json'
        filename_imglist_val = './datasets/kinetics_eval/tst2.json'
        filename_imglist_test = './datasets/kinetics_eval/tst3.json'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return n_class, filename_imglist_train, filename_imglist_val, filename_imglist_test, root_data

def return_dataset(dataset, modality):
    dict_single = {
    "KineticsZSAR":return_kineticszsar, 
    "KEVAL":return_keval
    }
    if dataset in dict_single:
        n_class, file_imglist_train, file_imglist_val, file_imglist_test, root_data = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    return n_class, file_imglist_train, file_imglist_val, file_imglist_test, root_data

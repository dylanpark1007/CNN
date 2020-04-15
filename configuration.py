
MR_len = 54
Subj_len = 115
Trec_len = 54
MPQA_len = 35
SST1_len = 100
SST2_len = 54
CR_len = 100

config = {
    'n_epochs' : 5,
    'kernel_sizes' : [3, 4, 5],
    'dropout_rate' : 0.5,
    'val_split' : 0.4,
    'edim' : 300,
    'n_words' : None,   #Leave as none
    'std_dev' : 0.05, #0.05(original) , 0.01
    'sentence_len' : Subj_len,
    'n_filters'  : 100,
    'batch_size' : 50,
    'dataset' : 'Subj',
    'model option' : 'static'

}

# sentence_len     MR = 54 , Subj = 115, Trec = 35, MPQA = 35 , SST-1 = 54 , SST-2 = 54 , CR = 100 (나머지 데이터셋 하나씩 최대문장길이 실험해보기)
# model option     rand, static, non-static, multichannel

# 1. sst1은 std_dev가 0.05일때 잘 나옴

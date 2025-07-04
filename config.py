# %%writefile /kaggle/working/-SLCL-Boundary/config.py
import numpy as np

MODEL = 'dr_unet'
BATCH_SIZE = 16
EVAL_BS = 32
NUM_WORKERS = 8

MOMENTUM = 0.9
NUM_CLASSES = 4

SAVE_PRED_EVERY = 50

# Hyper Paramters
WEIGHT_DECAY = 0.0001
LEARNING_RATE = 0.02 
LEARNING_RATE_DECAY = 2e-3
LEARNING_RATE_EPS = 20

POWER = 0.9
RANDOM_SEED = 1234

INPUT_SIZE = 224
DATA_DIRECTORY = "/kaggle/input/ct-mr-2d-dataset-da/CT_MR_2D_Dataset_mmwhs"
RAW_DATA_DIRECTORY = "/kaggle/input/ct-mr-2d-dataset-da/CT_MR_2D_Dataset_mmwhs"
EPOCHS = 2
WARMUP_EPOCHS = EPOCHS
EPS_ITERS = 5

WEIGHT_INTER_LOSS = 1.
WEIGHT_INTRA_LOSS = .1
# WEIGHT_INST_LOSS = 1.

CONTRASTIVE_LOSS_V = 'v1'
WEIGHT_MSE =0.002
WEIGHT_CONSIST = 2e-3
WEIGHT_EPS_CTS = .001
WEIGHT_THD = 0.
AUG_WEAK_MODE = 'simple' # e.g., 'simple', 'none' (for identity)
AUG_STRONG_MODE = 'heavy' # e.g., 'heavy', 'heavy2'

# Temperature for sharpening pseudo-labels (if used)
SHARPENING_TEMPERATURE = 0.5 
"""split 0"""
MMWHS_TEST_FOLD1 = [1, 4, 6, 7, 8, 9, 10, 11, 16, 17]
MMWHS_TEST_FOLD2 = [2, 3, 5, 12, 13, 14, 15, 18, 19, 20]

"""split 1"""
MMWHS_TEST_FOLD3 = [1, 4, 6, 7, 8, 10, 14, 15, 18, 19]
MMWHS_TEST_FOLD4 = [2, 3, 5, 9, 11, 12, 13, 16, 17, 20]

"""split 2"""
MMWHS_TEST_FOLD5 = [1, 3, 8, 9, 10, 12, 15, 16, 17, 18]
MMWHS_TEST_FOLD6 = [2, 4, 5, 6, 7, 11, 13, 14, 19, 20]

"""split 3"""
MMWHS_TEST_FOLD7 = [1, 3, 5, 6, 7, 8, 9, 10, 12, 19]
MMWHS_TEST_FOLD8 = [2, 4, 11, 13, 14, 15, 16, 17, 18, 20]

"""split 4"""
MMWHS_TEST_FOLD9 = [2, 4, 6, 7, 8, 9, 10, 11, 15, 18]
MMWHS_TEST_FOLD10 = [1, 3, 5, 12, 13, 14, 16, 17, 19, 20]

"""split 5"""
MMWHS_TEST_FOLD11 = [1, 2, 4, 6, 7, 8, 11, 12, 16, 19]
MMWHS_TEST_FOLD12 = [3, 5, 9, 10, 13, 14, 15, 17, 18, 20]

"""split 6"""
MMWHS_TEST_FOLD13 = [2, 5, 6, 8, 9, 10, 13, 14, 15, 17]
MMWHS_TEST_FOLD14 = [1, 3, 4, 7, 11, 12, 16, 18, 19, 20]

"""split 7"""
MMWHS_TEST_FOLD15 = [1, 2, 3, 4, 6, 7, 12, 13, 14, 18]
MMWHS_TEST_FOLD16 = [5, 8, 9, 10, 11, 15, 16, 17, 19, 20]

"""split 8"""
MMWHS_TEST_FOLD17 = [2, 3, 5, 6, 10, 11, 12, 16, 18, 19]
MMWHS_TEST_FOLD18 = [1, 4, 7, 8, 9, 13, 14, 15, 17, 20]

"""split 9"""
MMWHS_TEST_FOLD19 = [3, 5, 7, 10, 12, 13, 14, 16, 17, 20]
MMWHS_TEST_FOLD20 = [1, 2, 4, 6, 8, 9, 11, 15, 18, 19]

"""split 10"""
MMWHS_TEST_FOLD21 = [1, 2, 3, 5, 9, 10, 14, 15, 17, 19]
MMWHS_TEST_FOLD22 = [4, 6, 7, 8, 11, 12, 13, 16, 18, 20]

"""split 11"""
MMWHS_TEST_FOLD23 = [1, 2, 3, 5, 8, 12, 13, 16, 17, 20]
MMWHS_TEST_FOLD24 = [4, 6, 7, 9, 10, 11, 14, 15, 18, 19]

"""split 12"""
MMWHS_TEST_FOLD25 = [2, 3, 4, 5, 8, 12, 13, 16, 17, 20]
MMWHS_TEST_FOLD26 = [1, 6, 7, 9, 10, 11, 14, 15, 18, 19]

"""split 13; without sample 1"""
MMWHS_TEST_FOLD27 = [2, 3, 4, 5, 8, 12, 13, 16, 17, 20]
MMWHS_TEST_FOLD28 = [6, 7, 9, 10, 11, 14, 15, 18, 19]

MMWHS_3FOLD_0 = [5, 6, 8, 10, 11, 17, 18]
MMWHS_3FOLD_1 = [1, 9, 13, 14, 16, 19, 20]
MMWHS_3FOLD_2 = [2, 3, 4, 7, 12, 15]

MMWHS_TEST_FOLD00 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
MMWHS_TEST_FOLD01 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

train_extra_list = [[MMWHS_TEST_FOLD1, MMWHS_TEST_FOLD2], [MMWHS_TEST_FOLD3, MMWHS_TEST_FOLD4],
                    [MMWHS_TEST_FOLD5, MMWHS_TEST_FOLD6], [MMWHS_TEST_FOLD7, MMWHS_TEST_FOLD8],
                    [MMWHS_TEST_FOLD9, MMWHS_TEST_FOLD10], [MMWHS_TEST_FOLD11, MMWHS_TEST_FOLD12],
                    [MMWHS_TEST_FOLD13, MMWHS_TEST_FOLD14], [MMWHS_TEST_FOLD15, MMWHS_TEST_FOLD16],
                    [MMWHS_TEST_FOLD17, MMWHS_TEST_FOLD18], [MMWHS_TEST_FOLD19, MMWHS_TEST_FOLD20],
                    [MMWHS_TEST_FOLD21, MMWHS_TEST_FOLD22], [MMWHS_TEST_FOLD23, MMWHS_TEST_FOLD24],
                    [MMWHS_TEST_FOLD25, MMWHS_TEST_FOLD26], [MMWHS_TEST_FOLD27, MMWHS_TEST_FOLD28],
                    [MMWHS_TEST_FOLD00, MMWHS_TEST_FOLD01], [MMWHS_3FOLD_0, MMWHS_3FOLD_1, MMWHS_3FOLD_2]
                    ]

MMWHS_CT_T_VALID_SET = np.arange(1, 6)  # 1 to 5 (5 in total)
MMWHS_CT_S_TRAIN_SET = np.arange(1, 33)  # 6 to 32 (27 in total)
MMWHS_MR_T_VALID_SET = np.array([21, 22, 27, 30, 43])  # (5 in total)
MMWHS_MR_T_VALID_SET1 = np.arange(21, 26)  # (5 in total)
MMWHS_MR_S_TRAIN_SET = np.arange(21, 47)  # 26 to 46 (21 in total)

MSCMRSEG_TEST_FOLD1 = [23, 24, 29, 27, 34, 16, 25, 8, 22, 36, 35, 18, 30, 10, 39, 26, 41, 12, 38, 43]
MSCMRSEG_TEST_FOLD2 = [6, 7, 9, 11, 13, 14, 15, 17, 19, 20, 21, 28, 31, 32, 33, 37, 40, 42, 44, 45]
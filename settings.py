from pathlib import Path


# Only change these constants values
DATASET_BASE_PATH = Path("/hd1")
BASE_PATH_CODE = Path("/home/daniel/rsna-project")
#BASE_PATH = Path("/hd1/rsna-hemorrhage/genesis-brain-hemorrhage")
DATASET_NAME = "rsna"
# Only change these constants values


DATASET_DICOM_PATH = DATASET_BASE_PATH.joinpath(DATASET_NAME, "dicom")
DATASET_PNG_PATH = DATASET_BASE_PATH.joinpath(DATASET_NAME, "png") # "png"

DATASET_TEXT_PATH = BASE_PATH_CODE.joinpath("dataset-text-files")


TRAINED_MODELS_PATH = DATASET_BASE_PATH.joinpath(DATASET_NAME, "trained_models", "6labels")

CODE_PATH = BASE_PATH_CODE.joinpath("codes")
CODE_PATH_MAIN = CODE_PATH.joinpath("train-model-6classes")
MODEL_INFO_PATH = CODE_PATH_MAIN.joinpath("model-info")


# Neural networks settings
EPOCHS = 20
LEARNING_RATE = 0.0001  # 10**(-4)

# NN Image
BATCH_SIZE = 64  # 1024



# DATASET_DICOM_PATH = DATASET_BASE_PATH.joinpath(DATASET_NAME, "dicom")
# DATASET_PNG_PATH = Path("png") #DATASET_BASE_PATH.joinpath(DATASET_NAME, "super-tiny") # "png"

# DATASET_TEXT_PATH = Path("") #BASE_PATH_CODE.joinpath("dataset-text-files")


# TRAINED_MODELS_PATH = Path("trained_models") #DATASET_BASE_PATH.joinpath(DATASET_NAME, "trained_models", "6labels")

# CODE_PATH = BASE_PATH_CODE.joinpath("codes")
# CODE_PATH_MAIN = CODE_PATH.joinpath("train-model-6classes")
# MODEL_INFO_PATH = Path("") #CODE_PATH_MAIN.joinpath("model-info")
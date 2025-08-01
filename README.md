# GRPO_Finetuning_Models
Finetuning a Generative Pretrained Transformer Model (GPT2) with GRPO

Expanding on ch.6 code from: [https://github.com/rasbt/LLMs-from-scratch]. The code in this repository uses Group-Relative Policy Optimized fine tuning to get the trained classifier that determines if an input text is SPAM or NOT SPAM.

## Try it in the could with a Live demo on Hugging Face spaces:
Try it here @ [SoggyBurritos/Spam_Classifier_Agent](https://huggingface.co/spaces/SoggyBurritos/Spam_Classifier_Agent)

## Run on you local machine (GPU HIGHLY RECOMMENDED)
1. 'git clone' this repository.
    - ```git clone https://github.com/BrianP490/GRPO_Finetuning_Models.git```

2. Use conda to install the required environment to run the interactive python notebooks and python script files.
    - ```conda env create -f ./setup/dev_requirements.yaml```

3. Activate the environment in your terminal:
    - ```conda activate GRPO_env```

4. Run the main python script [grpo_gpt2_00_02.py](grpo_gpt2_00_02.py) with the environment on OR interactive python notebook file [grpo_gpt2_00_02.ipynb](grpo_gpt2_00_02.ipynb) with the kernel selected as the environment. If running the notebook, you may also have to install the ipykernel when promted.
    - ```python grpo_gpt2_00_02.py```

# Main Script Parameters:

You can customize the behavior of the training script by providing the following command-line arguments:

- --epochs

    - Type: int

    - Default: 8

    - Description: Number of training epochs to run.

- --learning_rate

    - Type: float

    - Default: 0.0003

    - Description: Learning rate used by the optimizer.

- --dataloader_batch_size

    - Type: int

    - Default: 64

    - Description: Batch size used by the dataloaders for training, validation, and testing.

- --dataloader_pin_memory

    - Type: action='store_false' (boolean flag)

    - Default: True (if flag is not present)

    - Description: Disable pinned memory in dataloaders (enabled by default). Include this flag to disable it.

- --dataloader_num_workers

    - Type: int

    - Default: 0

    - Description: Number of subprocesses to use for data loading.

- --batch_size

    - Type: int

    - Default: 1024

    - Description: Batch size for collecting experience during training. Note: This parameter is currently not used in this implementation.

- --gpt2_size

    - Type: str

    - Default: "gpt2-small (124M)"

    - Description: GPT-2 model variant to use (e.g., "gpt2-small (124M)", "gpt2-medium (355M)", "gpt2-large (774M)", "gpt2-xl (1558M)").

- --k_epochs

    - Type: int

    - Default: 32

    - Description: Number of GRPO update steps per epoch.

- --epsilon

    - Type: float

    - Default: 0.2

    - Description: Clipping parameter for policy updates.

- --beta_kl

    - Type: float

    - Default: 0.01

    - Description: Coefficient for KL divergence penalty in GRPO loss calculation.

- --entropy_coeff

    - Type: float

    - Default: 0.05

    - Description: Coefficient for entropy regularization to encourage exploration.

- --log_iterations

    - Type: int

    - Default: 2

    - Description: Frequency (in iterations) to log training progress.

- --eval_iterations

    - Type: int

    - Default: 2

    - Description: Frequency (in iterations) to evaluate the model.

- --gamma

    - Type: float

    - Default: 0.99

    - Description: Discount factor for future rewards.

- --num_envs

    - Type: int

    - Default: 16

    - Description: Number of parallel environments used for experience collection. Note: This parameter is currently not used in this implementation.

- --use_cuda

    - Type: action='store_true' (boolean flag)

    - Default: False (if flag is not present)

    - Description: Enable CUDA for training if available. Include this flag to enable CUDA.

- --device

    - Type: str

    - Default: "cpu"

    - Description: Device to use for training (e.g., "cpu", "cuda:0"). This parameter overrides the --use_cuda flag if specified.

- --save_model

    - Type: action='store_true' (boolean flag)

    - Default: False (if flag is not present)

    - Description: Save the trained model after training. Include this flag to enable model saving.

- --model_output_path

    - Type: str

    - Default: "models/Spam-Classifier-GPT2-Model.pt"

    - Description: File path to save the trained model. Parent directories will be created if they do not exist.


### Example Commands:
- ```python grpo_gpt2_00_02.py```
    - Uses the default settings to run the script.
    
- ```python grpo_gpt2_00_02.py --epochs=32 --k_epochs=128 --log_iterations=4 --eval_iterations=8 --save_model```
    - Explanation: Run for 32 epochs and 128 Group Relative Policy Optimization cycles. Log the average epoch statistics (e.g. loss, R1 Ratio, Entropy, etc.) results every 4 epochs. Evaluate the Policy under training every 8 epochs. Lastly, save the trained model. Uses the default save path. Uses default for everything else.
    
    
- ```python grpo_gpt2_00_02.py --epochs=32 --k_epochs=128 --use_cuda --save_model --model_output_path=models/first-trained-model.pt```
    - Explanation: Let the system detect if your system has GPU capabilities and has cuda available for training. During the training setup, the system dynamically sets the device variable for model training. Save the trained model using the specified location. Uses default for everything else.


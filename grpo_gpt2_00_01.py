# %% [markdown]
### grpo_gpt2_00_01.py
# Using GRPO to finetune the chat gpt2 model

# %% [markdown]
# ## Imports

# %%
from importlib.metadata import version
import torch, tiktoken, time, os
import torch.optim as optim
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import zipfile
from pathlib import Path
import pandas as pd
from torch.nn import Module # For type hinting
from typing import Tuple # Import Tuple for type hinting
from tiktoken import Encoding  # For type hinting
from utils import GPTModel, load_weights_into_gpt, download_and_load_gpt2

# pkgs = ["numpy", 
#         "tiktoken", 
#         "torch",
#         # "tensorflow", # For OpenAI's pretrained weights
#         "pandas"
#        ]
# for p in pkgs:
#     print(f"{p} version: {version(p)}")

# %% [markdown]
# ## Dataset Class

# %%
def prepare_datasets(data_file_path="./sms_spam_collection/SMSSpamCollection.tsv", sep="\t", header=None, column_names=["Label", "Text"], train_frac=0.7, validation_frac=0.15, store_directory="./sms_spam_collection/data_splits"):
    """This function prepares the train, test, and validation datasets from the original data.
    Code inspired from: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/ch06.ipynb
    Args:
        data_file_path (str): Path to the original dataset.
        sep (str): The separator used in the dataset file.
        column_names (list): A list of strings representing the column names used to read the dataset file using pandas.
        train_frac (float): The percentage of the data used for training.
        validation_frac (float): The percentage of the overall data used for the validation dataset.
        store_directory (str): The parent directory path that will contain the 3 datasets (train, test, and validation) .

    Returns:
        store_directory (str): The parent directory path containing the 3 datasets.
        """
    
    print(f"'prepare_datasets' function call: Using data_file_path='{data_file_path}' to find the original dataset.\n Using store_directory='{store_directory}' for the train, test, and validation dataset parent directory")

    # Construct the full paths for the output files
    train_csv_path = os.path.join(store_directory, "train.csv")
    test_csv_path = os.path.join(store_directory, "test.csv")
    validation_csv_path = os.path.join(store_directory, "validation.csv")

    if os.path.exists(train_csv_path) and os.path.exists(test_csv_path) and os.path.exists(validation_csv_path) :
        print(f"Train, Test, and Validation datasets detected in '{store_directory}', skipping generation")
    else:
        print(f"Datasets not found in '{store_directory}' or incomplete. Generating datasets...")

        df = pd.read_csv(data_file_path, sep=sep, header=header, names=column_names)

        # Count the instances of "spam"
        num_spam = df[df["Label"] == "spam"].shape[0]
        
        # Randomly sample "ham" instances to match the number of "spam" instances
        ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
        
        # Combine ham "subset" with "spam"
        balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

        balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

        # Shuffle the entire DataFrame
        balanced_df = balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)

        # Calculate split indices
        train_end = int(len(balanced_df) * train_frac)
        validation_end = train_end + int(len(balanced_df) * validation_frac)

        # Split the DataFrame
        train_df = balanced_df[:train_end]
        test_df = balanced_df[validation_end:]
        validation_df = balanced_df[train_end:validation_end]

        # Create the directory if it doesn't exist
        os.makedirs(store_directory, exist_ok=True) 

        train_df.to_csv(train_csv_path, index=None)
        test_df.to_csv(test_csv_path, index=None)
        validation_df.to_csv(validation_csv_path, index=None)

    print(f"'prepare_datasets' function returning: {store_directory} as parent directory.")
    
    return store_directory

# %%
class SpamDataset(Dataset):
    """Dataset class to turn text into tokenized inputs
    Original Code in: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/ch06.ipynb"""
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        try:
            self.data = pd.read_csv(csv_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {csv_file}")

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]    # For each row in the text section of the pandas data frame tokenize the text string(sentence); creates list of token IDs for each example/item of the text data
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

# %% [markdown]
# ## Data Pipeline

# %%
def initialize_datasets_and_dataloaders_pipeline(data_file_path=None, store_directory=None, batch_size=64, num_workers=0, pin_memory=False, drop_last=True) -> tuple[Dataset, Dataset, Dataset, DataLoader, DataLoader, DataLoader, Encoding]:
    """This pipeline does the following: calls function to prepare the train, test, and validation datasets, creates the dataloaders for each dataset, initializes the tokenizer, and returns them. Original code: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/ch06.ipynb
    Args:
        data_file_path (str): Path to the original dataset.
        sep (str): The separator used in the dataset file.
        column_names (list): A list of strings representing the column names used to read the dataset file using pandas.
        train_frac (float): The percentage of the data used for training.
        validation_frac (float): The percentage of the overall data used for the validation dataset.
        store_directory (str): The parent directory path that will contain the 3 datasets (train, test, and validation).
        batch_size (int): The dataloader's batch_size.
        num_workers (int): The dataloader's number of workers.
        pin_memory (bool): The dataloader's pin memory option.
        drop_last (bool): The dataloader's drop_last option.

    Returns: 
        train_dataset (Dataset): Dataset Class for the training dataset.
        test_dataset (Dataset): Dataset Class for the test dataset.
        validation_dataset (Dataset): Dataset Class for the validation dataset.
        train_dataloader (DataLoader): The train dataloader.
        test_dataloader (DataLoader): The test dataloader.
        validation_dataloader (DataLoader): The validation dataloader.
        tokenizer (Encoding): The tokenizer used to convert the text data into tokens.
    """
    tokenizer = tiktoken.get_encoding("gpt2")

    # Construct the arguments for prepare_datasets conditionally
    prepare_args = {}
    if data_file_path is not None:
        prepare_args['data_file_path'] = data_file_path
    if store_directory is not None:
        prepare_args['store_directory'] = store_directory

    # Prepare the 3 datasets files and return their parent directory
    store_directory = prepare_datasets(**prepare_args)

    train_dataset = SpamDataset(csv_file=os.path.join(store_directory, "train.csv"), tokenizer=tokenizer)
    test_dataset = SpamDataset(csv_file=os.path.join(store_directory, "test.csv"), tokenizer=tokenizer)
    validation_dataset = SpamDataset(csv_file=os.path.join(store_directory, "validation.csv"), tokenizer=tokenizer)

    print(f"Created Train dataset with '{len(train_dataset)}' samples")
    print(f"Created Test dataset with '{len(test_dataset)}' samples")
    print(f"Created Validation dataset with '{len(validation_dataset)}' samples")

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    print(f"Created Train Dataloader with '{len(train_dataloader)}' batches.")
    print(f"Created Test Dataloader with '{len(test_dataloader)}' batches.")
    print(f"Created Validation dataset with '{len(validation_dataloader)}' batches.")
    print(f"Each Dataloader has a batch_size of {batch_size}")
    
    return (train_dataset, test_dataset, validation_dataset, train_dataloader, test_dataloader, validation_dataloader, tokenizer)

# %% [markdown]
# ## Building Policies

# %%
def build_new_policy(base_config, chosen_model="gpt2-small (124M)", num_classes=2) -> GPTModel:
    """Build and load in the GPT2 model. Swap out the Head layer, and freeze up to the last Transformer module for transfer learning. Code Inspired from: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/ch06.ipynb
    Args:
        base_config (dict): The base configurations of the gpt2 model indicating vocab_size, context_length, drop_rate, and qkv_bias.
        chosen_model (str): The specific gpt2 model to construct.
        num_classes (int): The amount of classes in the classification task.
    Returns:
        model (GPTModel): The constructed Transformer model for classification."""
    
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    base_config.update(model_configs[chosen_model]) # Add the emb_dim, n_layers, and n_heads to the config

    model_size = chosen_model.split(" ")[-1].lstrip("(").rstrip(")")    # Extract the number of parameters from the chosen_model
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")
    
    print(f"Downloading weights of chosen model and loading them to initialized instance")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2_models_directory")

    model = GPTModel(base_config)

    load_weights_into_gpt(model, params)
    
    for param in model.parameters(): # Freeze model parameters
        param.requires_grad = False 

    # Unfreeze the last transformer block
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    # Unfreeze the final layer normalizing layer
    for param in model.final_norm.parameters():
        param.requires_grad = True

    model.out_head = torch.nn.Linear(in_features=base_config["emb_dim"], out_features=num_classes) # Reconfigure the output layer for the classification task
    return model

# %%
def build_old_policy(base_config, chosen_model="gpt2-small (124M)", num_classes = 2) -> GPTModel:
    """Construct the GPT2 model architecture without loading the weights. Code inspired from: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/ch06.ipynb
    Args:
        base_config (dict): The base configurations of the gpt2 model indicating vocab_size, context_length, drop_rate, and qkv_bias.
        chosen_model (str): The specific gpt2 model to construct.
        num_classes (int): The amount of classes in the classification task.
    Returns:
        model (GPTModel): The constructed Transformer model for classification."""
    
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    base_config.update(model_configs[chosen_model]) # Add the emb_dim, n_layers, and n_heads to the config

    model_size = chosen_model.split(" ")[-1].lstrip("(").rstrip(")")
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")
    model = GPTModel(base_config)

    model.out_head = torch.nn.Linear(in_features=base_config["emb_dim"], out_features=num_classes) # Reconfigure the output layer
    return model

# %% [markdown]
# ## Utility Functions

# %%
def calculate_discounted_rewards(predictions, batch_labels, gamma) -> torch.tensor:
    """A general function to calculate the discounted rewards for each of the model's trajectories. For this implementation, however, use non-discounted rewards.
    Args:
        predictions (torch.tensor): A flattened 1-d tensor containing the policy's predictions.
        batch_labels (torch.tensor): A flattened 1-d tensor containing the target predictions.
        gamma (float): The amount of discount to apply to future rewards [Not Used in this Implementation].
    Returns:
        disc_rewards (torch.tensor): The 1-d tensor containing the discounted rewards of each of the model's trajectories.
        """
    disc_rewards = (predictions == batch_labels).float()    # Simple comparison to evaluate rewards for each example; output a tensor of floats
    return disc_rewards

# %%
def log_epoch_stats(epoch, epoch_limit, total_loss, ratio, entropy) -> None:
    print(f"=====================  [Epoch ({epoch})]  =====================")
    print("Last k_epoch stats:")
    print(f"Loss: {total_loss:.7f} | Ratio: {ratio:.7f} | Entropy Term: {entropy:.7f}")
    print(f"===========================================================")

# %%
def evaluate_policy(Policy: Module, dataloader: DataLoader, current_epoch: int = None, max_epochs: int=None, device: str = 'cpu') -> float:
    """
    Evaluates the policy model (greedy version) on a given dataset.
    Args:
        Policy (Module): The Policy Model.
        dataloader (DataLoader): The dataloader to evaluate with.
        current_epoch (int): The current epoch [optional].
        max_epochs (int): The maximum number of epochs [optional].
        device (str): The device that the calculations will take place on.
    Returns:
        accuracy (float): The calculated accuracy.
    """
    Policy.eval()   # Turn off dropout layers and prevent grad tracking

    # Dataset check before continuing
    if len(dataloader.dataset) == 0: # Check the underlying dataset size
        print(f"Warning: Evaluation dataset is empty. Skipping accuracy calculation.")
        return float('nan')
    
    accuracy, num_correct, num_of_samples = 0.0, 0.0, 0.0

    Softmax_lyr = torch.nn.Softmax(dim=-1)  # The layer to transform the logits to probabilities
    
    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device) # Move the training data to the target device

            logits = Policy(batch_inputs)[:,-1,:]   # Get logits from model and only focus on the last iterations of each sample!!!
            # print(old_logits)
            
            classification_probabilities = Softmax_lyr(logits)
            class_predictions = torch.argmax(classification_probabilities, dim=-1).flatten()
            num_of_samples += batch_labels.size(0)
            num_correct += sum((class_predictions == batch_labels).float()).item()
    
    accuracy = num_correct/num_of_samples
    if current_epoch and max_epochs:   # If the function was called in the training loop
        print(f"===================  [Epoch ({current_epoch}/{max_epochs})]  ===================")
        print(f"Entire Validation Dataset Accuracy: {accuracy:.4f}| {num_correct} / {num_of_samples} samples")
        print(f"====================================================")

    else:   # If the function was called outside of the training loop
        print(f"===============================================")
        print(f"Entire Dataset Accuracy: {accuracy:.4f} | {num_correct} / {num_of_samples} samples")
        print(f"=====================================================")

            
    Policy.train()  # Set back to training mode 
    return accuracy

# %%
def simple_spam_classify_single(Policy: GPTModel, input_text: str, tokenizer: Encoding, device='cpu'):
    """Used to test the Policy with a single messages to classify it as SPAM or NOT SPAM.
    Args:
        Policy (GPTModel): The Policy that will classify the text
        input_text (list): The list containing the input texts.
        tokenizer (Encoding): The tokenizer that turns text into tokens to feed the policy.
        device (str): The device to run the Policy and input tokens to.
    """
    Policy.eval().to(device)
    Softmax_lyr = torch.nn.Softmax(dim=-1)

    tokenized_text = tokenizer.encode(input_text)
    torch_text=torch.tensor(tokenized_text).unsqueeze(0)    # turn into a tensor and add a batch dimension
    model_inputs = torch_text.to(device)
    # print(f"torch_text: {torch_text} | {torch_text.shape}")
    # print(f"model_inputs: {model_inputs} | {model_inputs.shape}")

    with torch.no_grad():
        logits = Policy(model_inputs)[:,-1,:]
        print(f"logits: {logits}")
        Class_probabilities = Softmax_lyr(logits)
    prediction = torch.argmax(input=Class_probabilities, dim=-1)
    # print(f"prediction: {prediction}")

    print("==================================================================")
    print(f"Classifiying the following text as [SPAM or NOT SPAM]:")
    print(f"'{input_text}'")
    print(f"Prediction ... [ => {'SPAM' if prediction.item() == 1 else 'NOT SPAM'} <= ]")
    print("==================================================================")

# %%
def simple_spam_classify_batch(Policy: GPTModel, input_text: list, tokenizer: Encoding, device='cpu'):
    """Used to test the Policy with a batch of messages to classify them as SPAM or NOT SPAM.
    Args:
        Policy (GPTModel): The Policy that will classify the text
        input_text (list): The list containing the input texts.
        tokenizer (Encoding): The tokenizer that turns text into tokens to feed the policy.
        device (str): The device to run the Policy and input tokens to.
    """
    Policy.eval().to(device)
    Softmax_lyr = torch.nn.Softmax(dim=-1)
    pad_token_id=50256

    tokenized_text = [
            tokenizer.encode(text) for text in input_text    # For each row in the text section of the pandas data frame tokenize the text string(sentence); creates list of token IDs for each example/item of the text data
        ]
    
    max_length = 0
    for encoded_text in tokenized_text:
        encoded_length = len(encoded_text)
        if encoded_length > max_length:
            max_length = encoded_length
    
    # Pad sequences to the longest sequence
    encoded_texts = [
        encoded_text + [pad_token_id] * (max_length - len(encoded_text))
        for encoded_text in tokenized_text
    ]
    
    torch_text=torch.tensor(encoded_texts, dtype=torch.long)    # Turn into a tensor
    model_inputs = torch_text.to(device)
    # print(f"torch_text: {torch_text} | {torch_text.shape}")
    with torch.no_grad():
        logits = Policy(model_inputs)[:,-1,:]
        print(f"logits: {logits}")
        Class_probabilities = Softmax_lyr(logits)
    predictions = torch.argmax(input=Class_probabilities, dim=-1)
    print(f"predictions: {predictions}")

    bundle = zip(input_text, predictions)
    print(f"input_text:{input_text} | pred: {predictions}")

    for i, (text_str, pred) in enumerate(bundle):
        print("==================================================================")
        print(f"Classifiying the following text:")
        print(f"[SPAM || NOT SPAM]: \n'{text_str}'")
        print(f"Prediction ... [ => {'SPAM' if pred == 1 else 'NOT SPAM'} <= ]")
        print("==================================================================")

# %% [markdown]
# ## Training Loop

# %%
def grpo_train(model_config: dict, train_dataloader: DataLoader, validation_dataloader: DataLoader, gpt2_size="gpt2-small (124M)", epochs=32, learning_rate=0.0003, batch_size=64, gamma=0.99, k_epochs=64, epsilon=0.2, beta_kl=0.01, max_grad_norm=0.5, entropy_coeff=0.05, log_iterations=10, eval_iterations=10, device="cpu", num_envs:int=None) -> GPTModel:
    """The Group-Relative Policy Optimization Training function.

    Args:
        model_config (dict): The base configurations for building the policies.
        train_dataloader (DataLoader): The dataloader for the training loop.
        validation_dataloader (DataLoader): The dataloader for the validation loop.
        gpt2_size (str): The GPT2 model and parameter choice (e.g. 'gpt2-small (124M)').
        epochs (int): The number of times the outer loop is performed.
        learning_rate (float): The hyperparameter that affects how much the model's parameters learn on each update iteration.
        batch_size (int): The number of training examples that the Old Policy gathers to perform a GRPO update [Not Used in this Implementation].
        gamma (float): The discount rate used to calculate discounted rewards.
        k_epochs (int): The number of inner iterations used to update Policy New.
        epsilon (float): The hyperparameter that affects the clipping of the 'R1_ratio'.
        beta_kl (float): The hyperparameter that adjusts how much the KL Divergence term affects the overall GRPO Loss.
        max_grad_norm (float): Used to promote numerical stability and prevent exploding gradients.
        entropy_coeff (float): The hyperparameter that adjusts how much the entropy term affects the overall GRPO Loss.
        log_iterations (int): Used to log information about the state of the New Policy.
        eval_iterations (int): Used to run an evaluation of the New Policy.
        device (str): The device that the model will be trained on.
        num_envs (int): The number of parallel training environments that will be used during training [Not Used in this Implementation].

    Returns: 
        Policy_New (GPTModel): The Trained Model in evaluation mode.
    """
    # print(f"Training Policy on {device} with {epochs} main epochs, {k_epochs} inner epochs, {learning_rate} learning rate, batch size={batch_size}, KL beta={beta_kl}, gamma={gamma}, epsilon={epsilon}, beta_kl={beta_kl}, max_grad_norm={max_grad_norm}, entropy_coeff={entropy_coeff}.")
    print(f"Training Policy on {device} with {epochs} main epochs, {k_epochs} inner epochs, {learning_rate} learning rate, KL beta={beta_kl}, epsilon={epsilon}, beta_kl={beta_kl}, max_grad_norm={max_grad_norm}, entropy_coeff={entropy_coeff}.")
    print(f"Using gpt2 size: '{gpt2_size}', logging every {log_iterations} epoch iterations, evaluating every {eval_iterations} epoch iterations.")

    Policy_New = build_new_policy(model_config, chosen_model=gpt2_size, num_classes=2).to(device)   # STEP 1 || 
    Policy_New.train()
    # Policy_New = torch.compile(Policy_New) # To reap efficiency benefits ; not working due to Triton dependency

    # STEPS 2 || For I iterations --> OMITTED
    # STEPS 3 || Initialize a reference model --> OMITTED

    optimizer = optim.Adam(params=Policy_New.parameters(), lr=learning_rate)

    Policy_Old = build_old_policy(model_config, chosen_model=gpt2_size, num_classes=2).to(device)
    Policy_Old.eval()
    # Policy_Old = torch.compile(Policy_Old)

    classifier_lyr = torch.nn.Softmax(dim=-1)   # For validation loop

    for epoch in tqdm(range(epochs), desc=f">>>>>>>>>>>>>>>>>>>>>\nMain Epoch (Outer Loop)", leave=True):     # STEP 4 || 
        # STEP 5 || Sample a batch D_b from D
        batch_inputs, batch_labels = next(iter(train_dataloader))
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device) # move the training data to the target device
        # print(f"batch_inputs shape: {batch_inputs.shape}")
        # print(f"batch_labels shape: {batch_labels.shape}")

        # STEP 6 || Update the old policy model PI old <- PI new
        Policy_Old.load_state_dict(Policy_New.state_dict())
        
        # STEP 7 || Collect a Batch of Experiences Using the Old Policy
        with torch.no_grad():
            old_logits = Policy_Old(batch_inputs)[:,-1,:]   # Get logits from model and only focus on the last iterations of each sample
            # print(old_logits)
            old_dist = torch.distributions.Categorical(logits=old_logits) # Create a distribution to sample from
            old_predictions = old_dist.sample() # Tensor of shape (batch_size,) ; list of predictions
            # print(f"old_predictions: \n{old_predictions[:10]}")
            # print(f"batch_labels True Values: \n{batch_labels[:10]}")
            old_log_probs = old_dist.log_prob(old_predictions)

        # STEP 8 || Calculate "Discounted" Rewards for completed trajectories
        discounted_rewards = calculate_discounted_rewards(old_predictions, batch_labels, gamma)    # Output is a 1-d Tensor with "discounted" rewards per each sample in batch
        # print("Calculated discounted returns")
        # STEP 9 || Calculate the Advantage for each Trajectory using normalization
        all_advantages_tensor = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        Policy_New.train()  # Prepare Policy NN for updates

        # STEP 10 || GRPO Optimization ---
        for k_epoch in tqdm(range(k_epochs), desc=f"Epoch {epoch+1}/{epochs} (Inner K-Epochs)", leave=True):
            print(f"===========================  [({k_epoch+1}/{k_epochs})]  ==========================\n")
            optimizer.zero_grad()   # Flush out all the accumulated gradients for the weights of the model-under-training!!!

            new_logits = Policy_New(batch_inputs)[:,-1,:]   # Get logits from model and only focus on the last iterations of each sample!!!
            new_dist = torch.distributions.Categorical(logits=new_logits)
            new_log_probs = new_dist.log_prob(old_predictions)  # Get the log probability of choosing the same action that the old policy took using the new distribution
            entropy = new_dist.entropy().mean() # Calculate entropy for regularization
            # print(f"Entropy of this k_epoch: {entropy}")
            
            R1_ratio = torch.exp(new_log_probs - old_log_probs)

            unclipped_surrogate = R1_ratio * all_advantages_tensor
            clipped_surrogate = torch.clamp(input=R1_ratio, min=1.0-epsilon, max=1.0+epsilon) * all_advantages_tensor
            # print(f"unclipped_surrogate: \n{unclipped_surrogate[:10]}\nclipped_surrogate: \n{clipped_surrogate[:10]}")
            
            policy_loss = -torch.min(unclipped_surrogate, clipped_surrogate).mean()

            # Calculate KL divergence per sample, then take the mean over the batch
            # Note: Reusing the calculated logits from STEP #7
            kl_div_per_sample = torch.distributions.kl.kl_divergence(p=new_dist, q=old_dist)
            kl_loss = kl_div_per_sample.mean() # Mean over the batch

            # Total Loss for GRPO
            total_loss = policy_loss + beta_kl * kl_loss - entropy_coeff * entropy
            # print(f"KL Divergence Average Loss: {kl_loss}")

            # STEP 11 || Policy Updates
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(Policy_New.parameters(), max_grad_norm)
            optimizer.step()    # Update policy parameters using gradient ascent
                

        # --- Logging and Evaluation ---
        if (epoch + 1) % log_iterations == 0:
            log_epoch_stats(epoch=epoch+1, epoch_limit=epochs, total_loss=total_loss.item(), ratio=R1_ratio.mean().item(), entropy=entropy)

        if (epoch + 1) % eval_iterations == 0:
            accuracy = evaluate_policy(Policy_New, validation_dataloader, current_epoch=epoch+1, max_epochs=epochs, device=device)
                

    Policy_New.eval()   # Change to eval mode for evaluation after training is complete

    return Policy_New # Return the trained policy

# %% [markdown]
# ## Main Loop


# %%
def main(args) -> int:
    print("SETTING UP FOR TRAINING")
    
    if args.device:     # Check if the user specified to use a CPU or GPU for training
        device = args.device
    else:
        if args.use_cuda:   # Check if the user wanted to use CUDA if available.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SAVE_LOCATION = "./models/Spam-Classifier-GPT2-Model.pt"   # Define the model path and name of the trained model weights

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    # --- Data Preparation Pipeline --- 
    train_dataset, test_dataset, validation_dataset, train_dataloader, test_dataloader, validation_dataloader, tokenizer = initialize_datasets_and_dataloaders_pipeline(data_file_path="./sms_spam_collection/SMSSpamCollection.tsv", store_directory="./sms_spam_collection/data_splits", num_workers=args.dataloader_num_workers, batch_size=args.dataloader_batch_size, pin_memory=args.dataloader_pin_memory)

    print("BEGINNING TRAINING SCRIPT")
    start_time=time.time()

    trained_policy = grpo_train(
        model_config=BASE_CONFIG,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        gpt2_size=args.gpt2_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size, # Significantly larger batch size recommended for stability
        gamma=args.gamma,
        k_epochs=args.k_epochs,
        epsilon=args.epsilon,
        beta_kl=args.beta_kl,
        entropy_coeff=args.entropy_coeff,
        log_iterations=args.log_iterations,
        eval_iterations=args.eval_iterations,
        device=device,
        num_envs=args.num_envs
    )
    end_time=time.time()

    # --- Calculate Training Time --- 

    elapsed_time= end_time - start_time
    hrs = int(elapsed_time / 3600)
    min = int((elapsed_time % 3600) / 60)
    seconds_remaining = elapsed_time - (hrs * 3600 ) - (min * 60)

    print(f"FINISHED MODEL TRAINING. \nTRAINING TOOK: {hrs} Hours, {min} Minutes, and {seconds_remaining:.3f} Seconds")

    # --- Testing Trained Model --- 
    print("\nTESTING THE TRAINED POLICY:")

    test_dataset_accuracy = evaluate_policy(trained_policy, test_dataloader, current_epoch=None, max_epochs=None, device=device)

    # ---  Saving Model Section  ---   

    if args.save_model:     # Check if the user wants to save the trained model weights
        if args.model_output_path:     # Check if the user specified a target save location
            SAVE_LOCATION=args.model_output_path

        parent_dir = os.path.dirname(SAVE_LOCATION)

        # If parent_dir is empty, it means the SAVE_LOCATION is just a filename
        # in the current directory, so no new directories need to be created.
        if parent_dir and parent_dir != '.':
            try:
                os.makedirs(parent_dir, exist_ok=True)
                print(f"Parent directory '{parent_dir}' created.")
            except OSError as e:
                print(f"Error creating directory {parent_dir}: {e}")
                # Depending on your application, you might want to exit or handle this more gracefully
                # For example, fall back to a default save location or skip saving.
        
        try:
            torch.save(trained_policy.state_dict(), f=SAVE_LOCATION)
            print(f"Model weights saved in: {SAVE_LOCATION}")
        except Exception as e:
            print(f"Error saving model to {SAVE_LOCATION}: {e}")

    return 0

# %%
# Example usage (assuming you have a way to call this function, e.g., in a main block)
if __name__ == '__main__':
    # --- Begin Timing Main Script Execution Time ---
    start_time=time.time()

    parser = argparse.ArgumentParser(description="Train and evaluate a GPT-based Spam Classifier using GRPO.")

    parser.add_argument('--epochs', type=int, default=8,
        help='(int, default=8) Number of training epochs to run.')

    parser.add_argument('--learning_rate', type=float, default=0.0003,
        help='(float, default=0.0003) Learning rate used by the optimizer.')

    parser.add_argument('--dataloader_batch_size', type=int, default=64,
        help='(int, default=64) Batch size used by the dataloaders for training, validation, and testing.')

    parser.add_argument('--dataloader_pin_memory', action='store_false',
        help='(bool, default=True) Disable pinned memory in dataloaders (enabled by default).')

    parser.add_argument('--dataloader_num_workers', type=int, default=0,
        help='(int, default=0) Number of subprocesses to use for data loading.')

    parser.add_argument('--batch_size', type=int, default=1024,
        help='(int, default=1024) Batch size for collecting experience during training. [Not Used in this Implementation]')

    parser.add_argument('--gpt2_size', type=str, default="gpt2-small (124M)",
        help='(str, default="gpt2-small (124M)") GPT-2 model variant to use (e.g., "gpt2-small (124M)", "gpt2-medium (355M)", "gpt2-large (774M)", "gpt2-xl (1558M)").')

    parser.add_argument('--k_epochs', type=int, default=32,
        help='(int, default=32) Number of GRPO update steps per epoch.')

    parser.add_argument('--epsilon', type=float, default=0.2,
        help='(float, default=0.2) Clipping parameter for policy updates.')

    parser.add_argument('--beta_kl', type=float, default=0.01,
        help='(float, default=0.01) Coefficient for KL divergence penalty in GRPO loss calculation.')

    parser.add_argument('--entropy_coeff', type=float, default=0.05,
        help='(float, default=0.05) Coefficient for entropy regularization to encourage exploration.')

    parser.add_argument('--log_iterations', type=int, default=2,
        help='(int, default=2) Frequency (in iterations) to log training progress.')

    parser.add_argument('--eval_iterations', type=int, default=2,
        help='(int, default=2) Frequency (in iterations) to evaluate the model.')

    parser.add_argument('--gamma', type=float, default=0.99,
        help='(float, default=0.99) Discount factor for future rewards.')

    parser.add_argument('--num_envs', type=int, default=16,
        help='(int, default=16) Number of parallel environments used for experience collection. [Not Used in this Implementation]')

    parser.add_argument('--use_cuda', action='store_true',
        help='(bool, default=False) Enable CUDA for training if available.')

    parser.add_argument('--device', type=str, default='cpu',
        help='(str, default="cpu") Device to use for training (e.g., "cpu", "cuda:0"). Overrides --use_cuda.')

    parser.add_argument('--save_model', action='store_true',
        help='(bool, default=False) Save the trained model after training.')

    parser.add_argument('--model_output_path', type=str, default='models/Spam-Classifier-GPT2-Model.pt',
        help='(str, default="models/Spam-Classifier-GPT2-Model.pt") File path to save the trained model.')

    # Parse the arguments
    args = parser.parse_args()

    print(args)

    ret = main(args)

    end_time=time.time()

    # --- Calculate Main Script Execution Time --- 

    elapsed_time= end_time - start_time
    hrs = int(elapsed_time / 3600)
    min = int((elapsed_time % 3600) / 60)
    seconds_remaining = elapsed_time - (hrs * 3600 ) - (min * 60)

    print(f"FINISHED MAIN SCRIPT\nOVERALL DURATION: {hrs} Hours, {min} Minutes, and {seconds_remaining:.3f} Seconds")
    if ret == 0:
        print("TERMINATING PROGRAM")
    else: 
        print("Main Scipt Error")

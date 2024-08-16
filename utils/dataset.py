import pandas as pd
from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, path_to_csv, split_type, tokenizer, max_length):
        """
        Args:
            path_to_csv (string): Path to the csv file with annotations
            split_type (string): One of 'train', 'val', or 'test' to select the data split
            tokenizer (Tokenizer): Tokenizer to use for SMILES
        """
        # Load the entire dataset
        self.full_dataset = pd.read_csv(path_to_csv)

        # Randomly sample 1000 rows from the dataset
        # self.full_dataset = self.full_dataset.sample(1000)

        # Filter the dataset based on the split type
        self.dataset = self.full_dataset[self.full_dataset['split'] == split_type]

        # Get spectrum columns
        self.spectrum_cols = [col for col in self.dataset.columns if col.startswith('spectrum_')]
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get spectrum
        spectrum = self.dataset.iloc[idx][self.spectrum_cols].values
        spectrum = torch.tensor(spectrum.astype('float32')).unsqueeze(0) # Add channel dimension so shape is 1 x L
        spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min()) # Normalize to [0, 1]

        # Get SMILES string
        smiles = self.dataset.iloc[idx]['smiles']
        smiles = self.tokenizer.tokenize(smiles, max_length=self.max_length)

        return spectrum, smiles

import torch
import re
from datasets import load_dataset
import string


class CharDataset(torch.utils.data.Dataset):
    """
    torch dataset class for character strings.
    input: 2D tensor of longs corresponding to tokenized text string, context length
    datapoints are context length substrings, and paired substring shifted by 1
    """
    def __init__(self, data, context_length):
        self.data = data
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length
    
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.context_length]
        y = self.data[idx + 1 : idx + self.context_length + 1]
        return x, y
    

def clean_char_text(text: str) -> str:
    """
    remove all special characters from string, reduce to lowercase + whitespace vocabulary
    """
    # lowercase
    text = text.lower()
    
    # keep only a–z and whitespace
    text = re.sub(r'[^a-z\s]', '', text)

    # collapse all whitespace to single spaces
    text = re.sub(r'\s+', ' ', text)

    # strip whitespace at boundary
    text = text.strip()
    
    return text


class CharTokenizer:
    def __init__(self):
        # unique characters
        self.unq_char = list(sorted(set(' ' + string.ascii_lowercase)))
        self.V = len(self.unq_char)

        # character to token dict
        self.char_to_id = {self.unq_char[i] : i for i in range(self.V)}
        # token to character dict
        self.id_to_char = {i : self.unq_char[i] for i in range(self.V)}

    def encode(self, text):
         # Convert text to list of IDs
        return torch.tensor([self.char_to_id[ch] for ch in text], dtype=torch.long)
    
    def decode(self, ids):
        #convert list of ids back to text
        return "".join([self.id_to_char[id] for id in ids])



def construct_and_save_dataset(context_length, path, min_length = 512):
    ds = load_dataset("rahular/simple-wikipedia")

    # take all text examples
    concat_string = ''
    for dat in ds['train']:
        if len(dat['text']) > min_length:
            concat_string += dat['text']

    clean_string = clean_char_text(concat_string)

    tokenizer = CharTokenizer()
    tokenized_dataset = tokenizer.encode(clean_string)

    dataset = CharDataset(tokenized_dataset, context_length)

    torch.save(dataset, path)
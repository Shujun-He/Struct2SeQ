import pickle
import numpy as np



import torch
from torch.utils.data import Dataset, DataLoader
from arnie.utils import convert_dotbracket_to_bp_list

# def create_ct_matrix(structure, allow_pseudoknots=True):
#     """
#     Create a contact (CT) matrix from the structure.
    
#     Args:
#     structure (str or torch.Tensor): The RNA structure
#     allow_pseudoknots (bool): Whether to allow pseudoknots in the structure
    
#     Returns:
#     torch.Tensor: The CT matrix
#     """
#     if isinstance(structure, torch.Tensor):
#         dbn = detokenize_dot_bracket(structure.numpy())
#     else:
#         dbn = structure

#     bps = convert_dotbracket_to_bp_list(dbn, allow_pseudoknots=allow_pseudoknots)
    
#     length = len(dbn)
#     ct_matrix = np.zeros((length, length), dtype='float32')
    
#     for i, j in bps:
#         ct_matrix[i, j] = ct_matrix[j, i] = 1
    
#     for i in range(length):
#         ct_matrix[i, i] = 1
    
#     return torch.tensor(ct_matrix), bps


class RNADataset(Dataset):
    def __init__(self, files):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'sequence' and 'structure' columns.
        """
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        #data_dict = self.data[idx]
        f = self.files[idx]
        data = pickle.load(open(f,'rb'))


        structure = data[0]
        sequence = data[1]
        reward = data[2]
        
        dbn=detokenize_dot_bracket(structure.numpy())
        bps = convert_dotbracket_to_bp_list(dbn, allow_pseudoknots=True)
        ct_matrix=np.zeros((len(structure),len(structure)),dtype='float32')

        for i,j in bps:
            ct_matrix[i,j]=ct_matrix[j,i]=1
        
        for i in range(len(structure)):
            ct_matrix[i,i]=1

        

        return {"sequence":sequence, 
                "structure":structure,
                "ct_matrix":ct_matrix,
                "reward":reward}


class DotBracketDataset(Dataset):
    def __init__(self, structures):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'sequence' and 'structure' columns.
        """
        self.structures = structures

    def __len__(self):
        return len(self.structures)

    def get_paired_correspondences(self,bp):
        paired_correspondences={}
        for i,j in bp:
            paired_correspondences[i]=j
            paired_correspondences[j]=i
        return paired_correspondences

    def __getitem__(self, idx):
        #data_dict = self.data[idx]

        structure = self.structures[idx]
        #structure = torch.tensor(tokenize_dot_bracket(structure))
        bps = convert_dotbracket_to_bp_list(structure, allow_pseudoknots=True)
        ct_matrix=np.zeros((len(structure),len(structure)))

        for i,j in bps:
            ct_matrix[i,j]=ct_matrix[j,i]=1
        
        for i in range(len(structure)):
            ct_matrix[i,i]=1

        PC = self.get_paired_correspondences(bps)
        src = torch.tensor(tokenize_dot_bracket(structure))
        ct_matrix = torch.tensor(ct_matrix)
        return {
                "src":src,
                "paired_correspondence":PC,
                "ct_matrix":ct_matrix}
    
def collate_dbn(data):
    #return None
    # print(data)
    # exit()
    src=[d['src'] for d in data]
    ct_matrix=[d['ct_matrix'] for d in data]
    paired_correspondence=[d['paired_correspondence'] for d in data]

    src=torch.stack(src)
    ct_matrix=torch.stack(ct_matrix).float()

    return {
                "src":src,
                "ct_matrix":ct_matrix,
                "paired_correspondence":paired_correspondence
               }

class RNASequenceDataset(Dataset):
    def __init__(self, sequences):
        """
        Initialize the dataset with a list of RNA sequences.
        
        Args:
        sequences (list of torch.Tensor): List of RNA sequences, each as a tensor.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.sequences = sequences

        # Ensure all sequences have the same length
        self.seq_length = sequences[0].size(0)
        assert all(seq.size(0) == self.seq_length for seq in sequences), "All sequences must have the same length"

    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieve a sequence by index.
        
        Args:
        idx (int): Index of the sequence to retrieve.
        
        Returns:
        torch.Tensor: The requested RNA sequence.
        """
        sequence = self.sequences[idx]
        
        
        return {"sequence":sequence}

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to stack sequences into a batch.
        
        Args:
        batch (list): List of sequences to be collated.
        
        Returns:
        torch.Tensor: Batched sequences.
        """
        return torch.stack(batch)

def tokenize_dot_bracket(dot_bracket_str):
    # Extended mapping of dot-bracket characters and upper/lower case alphabets to integers
    char_to_index = {
        '.': 0,  # Unpaired nucleotide
        '(': 1,  # Paired nucleotide, left side of pair
        ')': 2,  # Paired nucleotide, right side of pair
        '[': 3,  # Alternative pairing options
        ']': 4,
        '{': 5,
        '}': 6,
        '<': 7,
        '>': 8,
        'A': 9, 'a': 10,
        'B': 11, 'b': 12,
        'C': 13, 'c': 14,
        'D': 15, 'd': 16,
        'E': 17, 'e': 18,
        'F': 19, 'f': 20,
        'G': 21, 'g': 22,
        'H': 23, 'h': 24,
        'I': 25, 'i': 26,
        'J': 27, 'j': 28,
        'K': 29, 'k': 30,
        'L': 31, 'l': 32,
        'M': 33, 'm': 34,
        'N': 35, 'n': 36,
        'O': 37, 'o': 38,
        'P': 39, 'p': 40,
        'Q': 41, 'q': 42,
        'R': 43, 'r': 44,
        'S': 45, 's': 46,
        'T': 47, 't': 48,
        'U': 49, 'u': 50,
        'V': 51, 'v': 52,
        'W': 53, 'w': 54,
        'X': 55, 'x': 56,
        'Y': 57, 'y': 58,
        'Z': 59, 'z': 60
    }
    # Convert dot-bracket string to list of indices
    return [char_to_index[char] for char in dot_bracket_str if char in char_to_index]

def detokenize_dot_bracket(token_list):
    # Create an inverse mapping of integers to dot-bracket characters
    index_to_char = {
        0: '.',
        1: '(',
        2: ')',
        3: '[',
        4: ']',
        5: '{',
        6: '}',
        7: '<',
        8: '>',
        9: 'A', 10: 'a',
        11: 'B', 12: 'b',
        13: 'C', 14: 'c',
        15: 'D', 16: 'd',
        17: 'E', 18: 'e',
        19: 'F', 20: 'f',
        21: 'G', 22: 'g',
        23: 'H', 24: 'h',
        25: 'I', 26: 'i',
        27: 'J', 28: 'j',
        29: 'K', 30: 'k',
        31: 'L', 32: 'l',
        33: 'M', 34: 'm',
        35: 'N', 36: 'n',
        37: 'O', 38: 'o',
        39: 'P', 40: 'p',
        41: 'Q', 42: 'q',
        43: 'R', 44: 'r',
        45: 'S', 46: 's',
        47: 'T', 48: 't',
        49: 'U', 50: 'u',
        51: 'V', 52: 'v',
        53: 'W', 54: 'w',
        55: 'X', 56: 'x',
        57: 'Y', 58: 'y',
        59: 'Z', 60: 'z'
    }
    
    # Convert list of indices to dot-bracket string
    return ''.join(index_to_char[token] for token in token_list if token in index_to_char)


def tokenize_sequence(sequence_str):
    # ACGU mapping to integers
    nt_to_index = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    nt_to_index={nt:i for i,nt in enumerate('ACGU')}
    # Convert RNA sequence to list of indices
    return [nt_to_index[char] for char in sequence_str if char in nt_to_index]
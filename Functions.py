import numpy as np
import csv
from os import path
import polars as pl
import yaml
from arnie.utils import convert_dotbracket_to_bp_list, convert_bp_list_to_dotbracket
import os
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupScheduler(_LRScheduler):
    """Linear warmup learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        total_steps: Total number of steps in one epoch (len(train_loader))
        final_lr: Target learning rate at the end of warmup
        
    Note:
        Despite the name, self.last_epoch inherited from _LRScheduler 
        actually counts steps, not epochs. It starts at -1 and is 
        incremented by 1 every time scheduler.step() is called.
    """
    def __init__(self, optimizer, total_steps, final_lr):
        self.total_steps = total_steps
        self.final_lr = final_lr
        super().__init__(optimizer)  # last_epoch=-1 by default

    def get_lr(self):
        # self.last_epoch is actually the current step number (starts at 0)
        current_step = self.last_epoch
        # Calculate current step's learning rate
        progress = float(current_step) / self.total_steps
        # Clip progress to avoid lr going above final_lr
        progress = min(1.0, progress)
        
        return [self.final_lr * progress for _ in self.base_lrs]

#create dummy arnie config
# with open('arnie_file.txt','w+') as f:
#     f.write("linearpartition: . \nTMP: /tmp")
    
os.environ['ARNIEFILE'] = 'arnie_file.txt'

from arnie.pk_predictors import _hungarian

def standardize_dbn(s):
    bps=convert_dotbracket_to_bp_list(s,allow_pseudoknots=True)
    converted=convert_bp_list_to_dotbracket(bps,len(s))
    return converted

def cor2vec(corr,s):
    seq_len=len(s)
    vec=np.ones(seq_len)*-1

    for i in corr:
        j=corr[i]
        vec[i]=j
        vec[j]=i
    return vec


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

def write_config_to_yaml(config, file_path):
    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file)

def delete_modules(model):
    del model.encoder.encoder_layers.self_attn.in_proj_weight
    del model.encoder.encoder_layers.self_attn.in_proj_bias
    del model.encoder.encoder_layers.self_attn.out_proj.weight
    del model.encoder.encoder_layers.self_attn.out_proj.bias
    del model.encoder.encoder_layers.linear1.weight
    del model.encoder.encoder_layers.linear1.bias
    del model.encoder.encoder_layers.linear2.weight
    del model.encoder.encoder_layers.linear2.bias
    del model.encoder.encoder_layers.norm1.weight
    del model.encoder.encoder_layers.norm1.bias
    del model.encoder.encoder_layers.norm2.weight
    del model.encoder.encoder_layers.norm2.bias

    # Delete from decoder
    # del model.decoder.decoder_layers.self_attn.in_proj_weight
    # del model.decoder.decoder_layers.self_attn.in_proj_bias
    # del model.decoder.decoder_layers.self_attn.out_proj.weight
    # del model.decoder.decoder_layers.self_attn.out_proj.bias
    # del model.decoder.decoder_layers.multihead_attn.in_proj_weight
    # del model.decoder.decoder_layers.multihead_attn.in_proj_bias
    # del model.decoder.decoder_layers.multihead_attn.out_proj.weight
    # del model.decoder.decoder_layers.multihead_attn.out_proj.bias
    # del model.decoder.decoder_layers.linear1.weight
    # del model.decoder.decoder_layers.linear1.bias
    # del model.decoder.decoder_layers.linear2.weight
    # del model.decoder.decoder_layers.linear2.bias
    # del model.decoder.decoder_layers.norm1.weight
    # del model.decoder.decoder_layers.norm1.bias
    # del model.decoder.decoder_layers.norm2.weight
    # del model.decoder.decoder_layers.norm2.bias
    # del model.decoder.decoder_layers.norm3.weight
    # del model.decoder.decoder_layers.norm3.bias
    return model




def generate_sequence(model,src,target_correspondence,start_token=4,p=0.1):
    seq_len=len(src)
    model.eval()
    #p=0.1
    with torch.no_grad():
        src = src.unsqueeze(0)  # Add batch dimension if not present
        memory = model.encoder(src)

        tgt = torch.tensor([[start_token]], dtype=torch.long).cuda()  # Start token with batch dimension
        outputs = [start_token]

        for position in range(seq_len):
            tgt_mask = torch.triu(torch.full((tgt.size(1), tgt.size(1)), float('-inf'), device=tgt.device), diagonal=1)
            paired_encoding = (src==0).long()
            out = model.decoder(tgt, paired_encoding, memory, tgt_mask=tgt_mask)
            values = out[0, -1]
            # print(values)
            # break
            # Apply base pair constraints
            mask = create_base_pair_mask(position, target_correspondence, outputs)
            masked_values = values.masked_fill(mask.cuda(), float('-inf'))
            #masked_values = values
            #break
            # Sample from the policy, not really 
            if np.random.uniform()<p:
                probs = F.softmax(masked_values/10000000, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = torch.argmax(masked_values)
            
            outputs.append(next_token)
            tgt = torch.cat([tgt, torch.tensor([[next_token]], dtype=torch.long).cuda()], dim=-1)

    nts='ACGU'

    predicted_sequence = torch.tensor(outputs[1:])  
    return predicted_sequence

def generate_sequence_norules(model,src,target_correspondence,start_token=4,p=0.1):
    seq_len=len(src)
    model.eval()
    #p=0.1
    with torch.no_grad():
        src = src.unsqueeze(0)  # Add batch dimension if not present
        memory = model.encoder(src)

        tgt = torch.tensor([[start_token]], dtype=torch.long).cuda()  # Start token with batch dimension
        outputs = [start_token]

        for position in range(seq_len):
            tgt_mask = torch.triu(torch.full((tgt.size(1), tgt.size(1)), float('-inf'), device=tgt.device), diagonal=1)
            paired_encoding = (src==0).long()
            out = model.decoder(tgt, paired_encoding, memory, tgt_mask=tgt_mask)
            values = out[0, -1]
            # print(values)
            # break
            # Apply base pair constraints
            #mask = create_base_pair_mask(position, target_correspondence, outputs)
            #masked_values = values.masked_fill(mask.cuda(), float('-inf'))
            masked_values=values
            #masked_values = values
            #break
            # Sample from the policy, not really 
            if np.random.uniform()<p:
                probs = F.softmax(masked_values/10000000, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = torch.argmax(masked_values)
            
            outputs.append(next_token)
            tgt = torch.cat([tgt, torch.tensor([[next_token]], dtype=torch.long).cuda()], dim=-1)

    nts='ACGU'

    predicted_sequence = torch.tensor(outputs[1:])  
    return predicted_sequence

# def generate_sequence_batched(model, src_batch, target_correspondence_batch, start_token=4, p=0.1, max_len=None):
#     batch_size, seq_len = src_batch.shape
#     model.eval()
#     device = src_batch.device

#     with torch.no_grad():
#         memory = model.encoder(src_batch)

#         tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
#         outputs = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

#         if max_len is None:
#             max_len = seq_len

#         for position in range(max_len):
#             tgt_mask = torch.triu(torch.full((tgt.size(1), tgt.size(1)), float('-inf'), device=device), diagonal=1)
#             paired_encoding = (src_batch == 0).long()
#             out = model.decoder(tgt, paired_encoding, memory, tgt_mask=tgt_mask)
#             values = out[:, -1, :]

#             # Apply base pair constraints
#             #mask = create_base_pair_mask_batched(position, target_correspondence_batch, outputs)
#             mask = []
#             for i,tc in enumerate(target_correspondence_batch):
#                 mask = create_base_pair_mask(position, target_correspondence, outputs)
#                 mask.append(create_base_pair_mask(position, tc, outputs[i]))
#             mask = torch.stack(mask,0)
#             masked_values = values.masked_fill(mask, float('-inf'))

#             # Sample from the policy
#             random_sample = torch.rand(batch_size, device=device) < p
#             probs = F.softmax(masked_values / 1e7, dim=-1)
#             sampled_tokens = torch.multinomial(probs, 1)
#             max_tokens = torch.argmax(masked_values, dim=-1, keepdim=True)
#             next_tokens = torch.where(random_sample.unsqueeze(1), sampled_tokens, max_tokens)

#             outputs = torch.cat([outputs, next_tokens], dim=1)
#             tgt = torch.cat([tgt, next_tokens], dim=1)

#     return outputs[:, 1:]  # Remove start token




# def create_base_pair_mask_batch(position, target_correspondence, outputs, batch_size):
#     mask = torch.zeros(batch_size, 4, dtype=torch.bool, device=outputs.device)
#     for b in range(batch_size):
#         if position in target_correspondence[b]:
#             paired_position = target_correspondence[b][position]
#             if position > paired_position:
#                 left_base = outputs[b, paired_position + 1]
#                 if left_base == 0:  # A
#                     mask[b, 0:3] = True  # Only U is allowed
#                 elif left_base == 1:  # C
#                     mask[b, 0:2] = True  # Only G is allowed
#                     mask[b, 3] = True
#                 elif left_base == 2:  # G
#                     mask[b, 0] = True  # C and U are allowed
#                     mask[b, 2] = True
#                 elif left_base == 3:  # U
#                     mask[b, 1:4] = True  # Only A is allowed
#     return mask

# def generate_sequence_batched(model, src, target_correspondence, start_token=4, p=0.1):
#     batch_size, seq_len = src.shape
#     model.eval()
    
#     with torch.no_grad():
#         memory = model.encoder(src)

#         tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=src.device)
#         outputs = torch.full((batch_size, seq_len + 1), start_token, dtype=torch.long, device=src.device)

#         for position in range(seq_len):
#             out = model.decoder(tgt, memory)
#             values = out[:, -1, :]

#             mask = create_base_pair_mask_batch(position, target_correspondence, outputs, batch_size)
#             masked_values = values.masked_fill(mask, float('-inf'))

#             random_sample = torch.rand(batch_size, device=src.device) < p
#             probs = F.softmax(masked_values / 10000000, dim=-1)
#             sampled_tokens = torch.multinomial(probs, 1).squeeze(-1)
#             max_tokens = torch.argmax(masked_values, dim=-1)
            
#             next_tokens = torch.where(random_sample, sampled_tokens, max_tokens)
            
#             outputs[:, position + 1] = next_tokens
#             tgt = torch.cat([tgt, next_tokens.unsqueeze(1)], dim=1)

#     predicted_sequences = outputs[:, 1:]  # Remove start tokens
#     return predicted_sequences

def jaccard_similarity_base_pairs(bp_set1, bp_set2):
    """
    Compute Jaccard similarity between two sets of base pairs.
    
    :param bp_set1: List of base pair tuples for set 1
    :param bp_set2: List of base pair tuples for set 2
    :return: Jaccard similarity as a float between 0 and 1
    """
    # Convert lists of base pairs to sets of frozensets for efficient set operations
    set1 = set(frozenset(bp) for bp in bp_set1)
    set2 = set(frozenset(bp) for bp in bp_set2)
    
    # Compute intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # Compute Jaccard similarity
    jaccard = len(intersection) / len(union) if union else 1.0  # If both sets are empty, similarity is 1
    
    return jaccard

# Example usage
# bp_set1 = [[16, 43], [17, 44], [18, 41], [19, 42], [20, 39], [21, 40], [22, 37], [23, 38], [24, 35], [25, 36]]
# bp_set2 = [[16, 43], [17, 44], [18, 41], [19, 42], [20, 39], [21, 40], [22, 37], [23, 38], [26, 35], [27, 34]]


def generate_sequence_batched(model, src, target_correspondence, ct_matrix, start_token=4, p=0.1, max_len=None):
    _, seq_len = src.shape
    batch_size = len(target_correspondence)
    model.eval()
    
    with torch.no_grad():
        memory = model.encoder(src,ct_matrix)
        if len(memory)==1:
            memory = memory.expand(batch_size,*memory[0].shape)
        tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=src.device)
        outputs = torch.full((batch_size, 1), start_token, dtype=torch.long, device=src.device)
        A_cnt = np.zeros(len(target_correspondence))
        max_len = max_len or seq_len
        past_key_values = None
        total_values = torch.zeros(len(target_correspondence), device=src.device)
        for position in range(max_len):
            tgt_mask = torch.triu(torch.full((tgt.size(1), tgt.size(1)), float('-inf'), device=tgt.device), diagonal=1)
            paired_encoding = (src == 0).long()
            out, past_key_values = model.decoder(tgt, paired_encoding, memory, 
                                                        past_key_values = past_key_values,
                                                        tgt_mask=tgt_mask, use_cache=True)
            values = out[:, -1, :]

            # Apply base pair constraints
            mask = create_base_pair_mask_batched(position, target_correspondence, outputs, A_cnt, max_len)
            masked_values = values.masked_fill(mask, float('-inf'))

            # Sample from the policy
            random_sample = torch.rand(batch_size, device=src.device) < p
            probs = F.softmax(masked_values / 10000000, dim=-1)
            sampled_tokens = torch.multinomial(probs, 1)
            max_tokens = torch.argmax(masked_values, dim=-1, keepdim=True)
            next_tokens = torch.where(random_sample.unsqueeze(1), sampled_tokens, max_tokens)
            # print(values.shape)
            # print(next_tokens.shape)
            # #exit()
            # print(values.gather(-1, next_tokens).shape)
            # exit()
            total_values = total_values + values.gather(-1, next_tokens).squeeze(-1)

            outputs = torch.cat([outputs, next_tokens], dim=1)

            A_cnt = A_cnt + (next_tokens.squeeze()==0).cpu().float().numpy()
            tgt = torch.cat([tgt, next_tokens], dim=1)
    
    return outputs[:, 1:]#, total_values  # Remove start token


def generate_sequence_batched_sample(model, src, target_correspondence, ct_matrix, start_token=4, p=1.0, max_len=None):
    _, seq_len = src.shape
    batch_size = len(target_correspondence)
    model.eval()
    
    with torch.no_grad():
        memory = model.encoder(src,ct_matrix)
        if len(memory)==1:
            memory = memory.expand(batch_size,*memory[0].shape)
        tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=src.device)
        outputs = torch.full((batch_size, 1), start_token, dtype=torch.long, device=src.device)
        A_cnt = np.zeros(len(target_correspondence))
        max_len = max_len or seq_len
        past_key_values = None
        total_values = torch.zeros(len(target_correspondence), device=src.device)
        for position in range(max_len):
            tgt_mask = torch.triu(torch.full((tgt.size(1), tgt.size(1)), float('-inf'), device=tgt.device), diagonal=1)
            paired_encoding = (src == 0).long()
            out, past_key_values = model.decoder(tgt, paired_encoding, memory, 
                                                        past_key_values = past_key_values,
                                                        tgt_mask=tgt_mask, use_cache=True)
            values = out[:, -1, :]

            # Apply base pair constraints
            mask = create_base_pair_mask_batched(position, target_correspondence, outputs, A_cnt, max_len)
            masked_values = values.masked_fill(mask, float('-inf'))

            # Sample from the policy
            random_sample = torch.rand(batch_size, device=src.device) < p
            probs = F.softmax(masked_values, dim=-1)
            sampled_tokens = torch.multinomial(probs, 1)
            max_tokens = torch.argmax(masked_values, dim=-1, keepdim=True)
            next_tokens = torch.where(random_sample.unsqueeze(1), sampled_tokens, max_tokens)
            # print(values.shape)
            # print(next_tokens.shape)
            # #exit()
            # print(values.gather(-1, next_tokens).shape)
            # exit()
            total_values = total_values + values.gather(-1, next_tokens).squeeze(-1)

            outputs = torch.cat([outputs, next_tokens], dim=1)

            A_cnt = A_cnt + (next_tokens.squeeze()==0).cpu().float().numpy()
            tgt = torch.cat([tgt, next_tokens], dim=1)
    
    return outputs[:, 1:]#, total_values  # Remove start token

def generate_sequence_batched_accelerate(model, src, target_correspondence, ct_matrix, start_token=4, p=0.1, max_len=None):
    _, seq_len = src.shape
    batch_size = len(target_correspondence)
    model.eval()
    
    with torch.no_grad():
        memory = model.module.encoder(src,ct_matrix)
        if len(memory)==1:
            memory = memory.expand(batch_size,*memory[0].shape)
        tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=src.device)
        outputs = torch.full((batch_size, 1), start_token, dtype=torch.long, device=src.device)
        A_cnt = np.zeros(src.shape[0])
        max_len = max_len or seq_len
        past_key_values = None
        total_values = torch.zeros(len(target_correspondence), device=src.device)
        for position in range(max_len):
            tgt_mask = torch.triu(torch.full((tgt.size(1), tgt.size(1)), float('-inf'), device=tgt.device), diagonal=1)
            paired_encoding = (src == 0).long()
            out, past_key_values = model.module.decoder(tgt, paired_encoding, memory, 
                                                        past_key_values = past_key_values,
                                                        tgt_mask=tgt_mask, use_cache=True)
            values = out[:, -1, :]

            # Apply base pair constraints
            mask = create_base_pair_mask_batched(position, target_correspondence, outputs, A_cnt, max_len)
            masked_values = values.masked_fill(mask, float('-inf'))

            # Sample from the policy
            random_sample = torch.rand(batch_size, device=src.device) < p
            probs = F.softmax(masked_values / 10000000, dim=-1)
            sampled_tokens = torch.multinomial(probs, 1)
            max_tokens = torch.argmax(masked_values, dim=-1, keepdim=True)
            next_tokens = torch.where(random_sample.unsqueeze(1), sampled_tokens, max_tokens)
            # print(values.shape)
            # print(next_tokens.shape)
            # #exit()
            # print(values.gather(-1, next_tokens).shape)
            # exit()
            total_values = total_values + values.gather(-1, next_tokens).squeeze(-1)

            outputs = torch.cat([outputs, next_tokens], dim=1)

            A_cnt = A_cnt + (next_tokens.squeeze()==0).cpu().float().numpy()
            tgt = torch.cat([tgt, next_tokens], dim=1)
    
    return outputs[:, 1:]#, total_values  # Remove start token





def create_base_pair_mask(position, target_correspondence, outputs, A_cnt, max_len):
    mask = torch.zeros(4, dtype=torch.bool)  # Default to no mask (all False)
    if position in target_correspondence:
        paired_position = target_correspondence[position]
        if position > paired_position:
            # We're on the right side of the pair, so we need to mask based on the left side
            left_base = outputs[paired_position+1]
            #ACGU
            if left_base == 0:  # A
                mask[0:3] = True  # Only U is allowed
            elif left_base == 1:  # C
                mask[0:2] = True  # Only G is allowed
                mask[3] = True
            elif left_base == 2:  # G
                mask[0] = True  # C and U are allowed
                mask[2] = True
            elif left_base == 3:  # U
                mask[1:4] = True  # Only A is allowed
    #repeat constraint
    #The OpenTB constraints are no more than 3G, 4C, 4A and <40% adenine.

    max_repeat=[4,4,3,5] #ACGU
    if position>5:
        
        for nt in range(4):
            last=outputs[-max_repeat[nt]:]
            if (last==nt).sum()==max_repeat[nt]:
                mask[nt] = True
                break
        if mask.sum()==4:
            mask[nt] = False

    # print(A_cnt)
    # print(max_len)
    if A_cnt>max_len*0.4:
        mask[0]=True
        if mask.sum()==4:
            mask[0] = False


    return mask

def create_base_pair_mask_batched(position, target_correspondence, outputs, A_cnt, max_len):
    batch_size = outputs.shape[0]
    masks = []
    

    # print(len(target_correspondence))
    # print(len(outputs))
    # print(len(A_cnt))
    # exit()
    for i in range(batch_size):
        mask = create_base_pair_mask(position, target_correspondence[i], outputs[i], A_cnt[i], max_len)
        masks.append(mask)
    
    return torch.stack(masks).to(outputs.device)

import csv
def log_rewards(file_path, episode, train_rewards, test_rewards):
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode, train_rewards, test_rewards])

def generate_sequence_topk(model, src, target_correspondence, ct_matrix, start_token=4, k=5, max_len=None):
    seq_len = src.shape[1]
    model.eval()
    
    with torch.no_grad():
        memory = model.encoder(src, ct_matrix)
        _,L,C=memory.shape
        # print(L,C)
        # exit() 
        tgt = torch.full((1,1), start_token, dtype=torch.long, device=src.device)
        candidates = [{'sequence': tgt[0], 'previous_value':0., 'logits': 0., 'A_cnt': 0.}]

        max_len = max_len or seq_len
        gamma = torch.linspace(0.25, 0.0, max_len)
        for position in range(max_len):
            # Stack all candidate sequences
            tgt = torch.stack([c['sequence'] for c in candidates])
            # print(tgt.shape)
            # exit()
            cumulative_logits = torch.tensor([c['logits'] for c in candidates], device=src.device)

            # Create tgt_mask for all candidates
            tgt_mask = torch.triu(torch.full((tgt.size(1), tgt.size(1)), float('-inf'), device=tgt.device), diagonal=1)
            #tgt_mask = tgt_mask.unsqueeze(0).expand(tgt.size(0), -1, -1)

            # Expand src and memory for batch processing
            # print(src.shape)
            # print(tgt.shape)
            src_expanded = src.expand(tgt.size(0), -1)
            memory_expanded = memory.expand(tgt.size(0), L, C)
            paired_encoding = (src_expanded == 0).long()

            # Process all candidates in parallel
            out = model.decoder(tgt, paired_encoding, memory_expanded, tgt_mask=tgt_mask)
            # print(out.shape)
            # exit()
            values = out[:, -1, :]

            # Apply base pair constraints
            A_cnt=[a['A_cnt'] for a in candidates]
            mask = create_base_pair_mask_batched(position, [target_correspondence[0]]*len(tgt), tgt, A_cnt, max_len)
            masked_values = values.masked_fill(mask, float('-inf'))

            # print(masked_values.shape)
            # exit()

            # Consider all 4 RNA nucleotides for all candidates
            # new_logits = masked_values + cumulative_logits.unsqueeze(1)
            # top_values, top_indices = torch.topk(new_logits.view(-1), k)

            # Create new candidates
            # print(candidates)
            # exit()
            new_candidates = []
            for i in range(len(candidates)):
                #for value, index in zip(top_values, top_indices):
                sequence, previous_value, logits, A_cnt = candidates[i]['sequence'], candidates[i]['previous_value'], candidates[i]['logits'], candidates[i]['A_cnt']



                for nucleotide_idx in range(4):
                    #candidate_idx = index // 4
                    #nucleotide_idx = index % 4
                    new_sequence = torch.cat([sequence, 
                                            torch.tensor([nucleotide_idx], device=tgt.device)])
                    new_candidates.append({'sequence': new_sequence, 
                                            'previous_value': masked_values[i,nucleotide_idx].item(),
                                            'logits': logits+masked_values[i,nucleotide_idx].item(),
                                            'A_cnt': A_cnt+(nucleotide_idx==0)})
#+(1-gamma[position])*masked_values[i,nucleotide_idx].item()
            candidates = sorted(new_candidates, key=lambda x: x['logits'], reverse=True)[:k]
            #print(k)
            #print(candidates)
            #exit()
        #     print(len(candidates))
        # exit()
        # Stack all final candidate sequences
        top_k_sequences = torch.stack([c['sequence'] for c in candidates])
    # print(top_k_sequences.shape)
    # exit()
    return top_k_sequences[:, 1:]  # Remove start token and return k x L tensor


def bps2set(bps):
    return set([tuple(bp) for bp in bps])

def get_rescue_sequences(target_dbn,design_dbn,design_sequence):

    target_bps=bps2set(convert_dotbracket_to_bp_list(target_dbn,allow_pseudoknots=True))
    design_bps=bps2set(convert_dotbracket_to_bp_list(design_dbn,allow_pseudoknots=True))

    paired_bp_candidates = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']

    unpaired_bp_candidates = ['AA', 'AG', 'AC', 'UU', 'UC', 'GA', 'GG', 'CA', 'CU', 'CC']

    #generate new candidates
    missing_bps=[bp for bp in target_bps if bp not in design_bps]
    extra_bps=[bp for bp in design_bps if bp not in target_bps]

    # missing_bps.append((30,31))
    # extra_bps.append((2,32))

    # print(missing_bps)
    # print(extra_bps)
    # exit()

    if (len(missing_bps)+len(extra_bps))>3:
        return []

    candidates=[]


    for i,j in missing_bps:
        new_candidates=[]
        if len(candidates)>0:
            for c in candidates:
                for pairs in paired_bp_candidates:
                    new_c=c[:]
                    new_c.append((i,j,pairs))
                    new_candidates.append(new_c)
        else:
            for pairs in paired_bp_candidates:
                new_c=[]
                new_c.append((i,j,pairs))
                new_candidates.append(new_c)
        
        candidates=new_candidates

    for i,j in extra_bps:

        new_candidates=[]
        if len(candidates)>0:
            for c in candidates:
                for pairs in unpaired_bp_candidates:
                    new_c=c[:]
                    new_c.append((i,j,pairs))
                    new_candidates.append(new_c)
        else:
            for pairs in unpaired_bp_candidates:
                new_c=[]
                new_c.append((i,j,pairs))
                new_candidates.append(new_c)
        
        candidates=new_candidates


    design_sequence=list(design_sequence)
    new_sequences=[]
    for c in candidates:
        new_sequence=design_sequence[:]
        for i,j,pair in c:
            new_sequence[i]=pair[0]
            new_sequence[j]=pair[1]
        new_sequences.append("".join(new_sequence))

    new_sequences=list(set(new_sequences))

    return new_sequences

def hamming_distance(str1, str2):
    # Ensure strings are of the same length
    if len(str1) != len(str2):
        raise ValueError("Strings must be of the same length to compute Hamming distance.")
    
    # Calculate the Hamming distance
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

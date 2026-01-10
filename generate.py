import torch
#from Action import take_action
from Functions import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from Dataset import *
from Encoder_Decoder import *
import os
import argparse
from search_v2 import *
import time 

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID to use')
parser.add_argument('--n_structures', type=int, default=100)
parser.add_argument('--target_df', type=str, required=True, help='Path to the target CSV file which has Title and Dot-bracket columns')
parser.add_argument('--out_folder', type=str, default='results', help='Output folder to save results')
parser.add_argument('--output_csv', type=str, required=True, help='Path to save the output CSV file', default='output.csv')
parser.add_argument('--up_bias', type=float, required=True, help='how much to bias generation towards WT sequence')
parser.add_argument('--weights_path', type=str, required=True, help='weights of Struct2SeQ model')
parser.add_argument('--rnet_weights', type=str, required=True, help='weights of rnet model')
parser.add_argument('--rnet_ss_weights', type=str, required=True, help='weights of rnet-ss model')


args = parser.parse_args()

start_time = time.time()



# Set GPU ID environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_id))
os.makedirs(args.out_folder, exist_ok=True)



db_vocab_size = 100
rna_vocab_size = 5
embed_size = 384
nhead = 16
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = embed_size*4
dropout = 0.1
policy_network = DotBracketRNATransformer(db_vocab_size, rna_vocab_size, embed_size, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout).cuda()



delete_modules(policy_network)



from Env import DQN_env
env=DQN_env(args.rnet_weights,args.rnet_ss_weights)


p=0.1
gamma=0.5
k=16
L=100
episodes=50
episode_length=1
bs=60
data=[]



from Dataset import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
#from ranger import Ranger








policy_network.load_state_dict(torch.load(args.weights_path))





# In[88]:


test_df=pl.read_csv(args.target_df)
#test_df=test_df.filter(test_df['Source']=='Father Z')
test_df

test_structures=test_df['Dot-bracket'].to_list()
test_structures=[standardize_dbn(s) for s in test_structures]
test_structures_tensor=[torch.tensor(tokenize_dot_bracket(t)).cuda().long() for t in test_df['Dot-bracket']]
test_correspondences=[env.get_paired_correspondences(convert_dotbracket_to_bp_list(s,allow_pseudoknots=True)) 
                        for s in test_df['Dot-bracket']]

test_ct=[]
for s in test_df['Dot-bracket']:
    #s = standardize_dbn(s)
    bps = convert_dotbracket_to_bp_list(s, allow_pseudoknots=True)
    ct_matrix=np.zeros((len(s),len(s)),dtype='float32')

    for i,j in bps:
        ct_matrix[i,j]=ct_matrix[j,i]=1
    
    for i in range(len(s)):
        ct_matrix[i,i]=1
    test_ct.append(ct_matrix)

n_test=len(test_structures_tensor)

i=0
sequences=[]
structures=[]
str_sequences=[]
target_structures=[]
shape_profiles=[]
ids=[]
bps=[]
jacc=[]
#for i in tqdm(range(n_test)):
policy_network.eval()
nts='ACGU'
repeat=128
all_jaccard=[]
predicted_structures=[]
for i in tqdm(range(n_test)):
#for i in tqdm(range(1)):
    src=test_structures_tensor[i].repeat(repeat).reshape(repeat,-1)
    #ct=test_ct[i].repeat(repeat).reshape(repeat,*test_ct[i].shape)
    ct=[test_ct[i] for _ in range(repeat)]
    ct=np.stack(ct,0)
    ct=torch.tensor(ct).cuda()
    # print(ct.shape)
    # exit()

    wildtype_sequence=test_df['wild_type_sequence'][i]

    #randomly pick up bias between 0.75 to 1.0
    #up_bias = np.random.uniform(0.75,1.0)
    print(f"up bias: {args.up_bias}")
    weight, bias = make_ref_upweighted_params(wildtype_sequence, up_bias=args.up_bias)
    weight=weight.cuda()
    bias=bias.cuda()
    # print(weight.shape, bias.shape)
    # exit()

    sequence1=generate_sequence_batched(policy_network,src,[test_correspondences[i]]*repeat,ct,p=0.05, weight=weight, bias=bias)
    sequence2=generate_sequence_batched(policy_network,src,[test_correspondences[i]]*repeat,ct,p=0.1, weight=weight, bias=bias)
    sequence3=generate_sequence_batched_sample(policy_network,src,[test_correspondences[i]]*repeat,ct,p=1, weight=weight, bias=bias)
    #sequence4=generate_sequence_batched_sample(policy_network,src,[test_correspondences[i]]*repeat,ct,p=1)
    # if args.gpu_id=='0':
    #     topk_sequence=generate_sequence_topk(policy_network,src[0,None],[test_correspondences[i]]*repeat,ct[0,None],k=repeat)
    # # print(topk_sequence.shape)
    # # exit()
        # sequence=torch.cat([sequence1,
        #                     sequence2,
        #                     sequence3,
        #                     topk_sequence],0)
    # else:
    sequence=torch.cat([sequence1,
                        sequence2,
                        sequence3],0)
    #sequence=sequence[:2]
    #sequence=sequence1
    #structures_,bps_=env.get_structure(sequence.cuda())
    batch_size = 32
    n_samples = len(sequence)
    n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
    structures_,bps_=[],[]
    shape=[]
    for b in range(n_batches):
        structures,bps=env.get_structure(sequence[b*batch_size:(b+1)*batch_size])
        shape_profiles.extend(env.get_SHAPE(sequence[b*batch_size:(b+1)*batch_size]))
        structures_.extend(structures)
        bps_.extend(bps)
    #shape=np.concatenate(shape,0)
    #exit()
    str_sequences_=["".join([nts[i.item()] for i in s]) for s in sequence]
    str_sequences.extend(str_sequences_)
    predicted_structures.extend(structures_)
    ids.extend([test_df['Title'][i]]*len(sequence))
    target_structures.extend([test_df['Dot-bracket'][i]]*len(sequence))
    #exit()
    jacc_sequence=[jaccard_similarity_base_pairs(b,convert_dotbracket_to_bp_list(test_df['Dot-bracket'][i],allow_pseudoknots=True)) for b in bps_]
    all_jaccard.extend(jacc_sequence)

    best=np.argmax(jacc_sequence)
    best_jacc=jacc_sequence[best]
    best_sequence=str_sequences_[best]
    best_structure=structures_[best]
    best_bps=bps_[best]

    #best_sequence

    #rescue
    if max(jacc_sequence)!=1:
        target_bps=bps2set(convert_dotbracket_to_bp_list(test_structures[i],allow_pseudoknots=True))
        target_corr=env.get_paired_correspondences(convert_dotbracket_to_bp_list(test_structures[i],allow_pseudoknots=True))
        best_corr=env.get_paired_correspondences(convert_dotbracket_to_bp_list(best_structure,allow_pseudoknots=True))

        target_vec=cor2vec(target_corr,best_structure)
        best_vec=cor2vec(best_corr,best_structure)

        diff_pos=np.where(target_vec!=best_vec)[0].tolist()
        #diff_pos=list(diff_pos)
        #exit()
        print(f"diff_pos {diff_pos}")
        if len(diff_pos)<=4:
            new_sequences=get_mutated(best_sequence,diff_pos,[])
            src=[]
            for seq in new_sequences: 
                src.append(tokenize_sequence(seq))

            src=np.array(src)
            src=torch.tensor(src).cuda()

            batch_size = 32
            n_samples = len(src)
            n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
            structures_,bps_=[],[]
            for b in range(n_batches):
                structures,bps=env.get_structure(src[b*batch_size:(b+1)*batch_size])
                shape_profiles.extend(env.get_SHAPE(src[b*batch_size:(b+1)*batch_size]))
                structures_.extend(structures)
                bps_.extend(bps)
            
            jacc_rescue=[jaccard_similarity_base_pairs(b,list(target_bps)) for b in bps_]
        # best_jaccard.append(max(jacc_sequence))

            best_rescue_index=np.argmax(jacc_rescue)
            if jacc_rescue[best_rescue_index]>jacc_sequence[best]:
                print(f"better sequence found w rescue at jaccard: {jacc_rescue[best_rescue_index]}")


                best_jacc=jacc_rescue[best_rescue_index]
                best_sequence=new_sequences[best_rescue_index]
                best_structure=structures_[best_rescue_index]
                best_bps=bps_[best_rescue_index]

            str_sequences.extend(new_sequences)
            predicted_structures.extend(structures_)
            ids.extend([test_df['Title'][i]]*len(new_sequences))
            target_structures.extend([test_structures[i]]*len(new_sequences))
            all_jaccard.extend(jacc_rescue)
            
    sequences.append(best_sequence)
    structures.append(best_structure)
    bps.append(best_bps)
    jacc.append(best_jacc)
    #jacc.append(np.mean(jacc_sequence))

    # str_sequence=[nts[i.item()] for i in best_sequence]
    # str_sequence="".join(str_sequence)

    print("#####")
    print(test_df['Title'][i])
    print(f'Target vs DQN designed jaccard={best_jacc}')
    print(test_structures[i])
    print(best_structure)
    distance=hamming_distance(test_df["wild_type_sequence"][i],best_sequence)
    print(f'Target vs DQN sequence hammin distance={distance}')
    print(test_df["wild_type_sequence"][i])
    print(best_sequence)
    #break
print("best jaccrd",np.mean(np.array(jacc)))
solved=(np.array(jacc)==1.0).sum()
print(f"{solved}/{len(jacc)} solved")

#exit()
#convert shape_profiles to strings 
shape_profiles2save = [str(list([float(x) for x in profile])) for profile in shape_profiles]
df = pl.DataFrame({
    'sequence': str_sequences,
    'predicted_structure': predicted_structures,
    'source': ids,
    'shape_profile': shape_profiles2save,
    'target_structure': target_structures,
    'jaccard': all_jaccard
})

# Add additional computed columns
df = df.with_columns([
    #pl.col('sequence').str.lengths().alias('sequence_length'),
    #pl.col('predicted_structure').str.lengths().alias('predicted_structure_length'),
    #pl.col('target_structure').str.lengths().alias('target_structure_length'),
    (pl.col('predicted_structure') == pl.col('target_structure')).alias('structure_match')
])


df.write_csv(os.path.join(args.out_folder, args.output_csv))

time_elapsed = time.time() - start_time

print(f"Time elapsed: {time_elapsed} seconds")
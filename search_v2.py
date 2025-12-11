from itertools import product

# PKB129
# Target vs DQN designed jaccard=0.9545454545454546
# ..((....[[[[[[..((((((((((((.....))))))))))))...[[..))]].....]]]]]].
# ..((....[[[[[[..((((((((((((.....))))))))))))...[...)).].....]]]]]].
# Target vs DQN sequence hammin distance=51
# GCGUAAAUGUCGACUUGGAGGUUGUGCCCUUGAGGCGUGGCUUCCGGAGCUAACGCGUUAAGUCGACC
# AACCAAAACAUAUCCAACCUCACUCCCCAAAACGGGGAGUGAGGUAAACCAAGGGGAAACAGAUAUGC

from itertools import product

def edit_sequence(sequence, position, char):
    return sequence[:position]+char+sequence[position+1:]

def get_diff_positions(string1, string2):
    """
    Return list of positions where strings differ.
    
    Args:
        string1 (str): First string
        string2 (str): Second string
        
    Returns:
        list: List of 0-based positions where strings differ
    """
    if len(string1) != len(string2):
        raise ValueError(f"Strings must be same length. Got lengths {len(string1)} and {len(string2)}")
    
    return [i for i, (c1, c2) in enumerate(zip(string1, string2)) if c1 != c2]

target="..((....[[[[[[..((((((((((((.....))))))))))))...[[..))]].....]]]]]]."
design="..((....[[[[[[..((((((((((((.....))))))))))))...[...)).].....]]]]]]."

sequence="AACCAAAACAUAUCCAACCUCACUCCCCAAAACGGGGAGUGAGGUAAACCAAGGGGAAACAGAUAUGC"

# positions=get_diff_positions(target, design)


# current_sequences=[]
def get_mutated(sequence,positions,current_sequences):
    nts='AUGC'
    #print(current_sequences)
    #print(positions)
    new_sequences=[]
    if len(current_sequences)==0:
        current_sequence=sequence

        mutated_posiiton=positions.pop()
        for nt in nts:
            new_sequences.append(edit_sequence(current_sequence, mutated_posiiton, nt))

    else:
        mutated_posiiton=positions.pop()
        for current_sequence in current_sequences:
            for nt in nts:
                new_sequences.append(edit_sequence(current_sequence, mutated_posiiton, nt))

    current_sequences=new_sequences#[:]
    #print(current_sequences)
    if len(positions)==0:

        return current_sequences
    else:
        return get_mutated(sequence,positions,current_sequences)

    

# sequences=get_mutated(sequence,get_diff_positions(target, design),[])

# # def get_rescue_candidates(sequence, target, design):
# #     return get_mutated(sequence,get_diff_positions(target, design),[])

# for s in sequences:
#     print(s)

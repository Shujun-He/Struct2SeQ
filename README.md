# Struct2SeQ: RNA Inverse Folding with Deep Q-Learning

**Inference Code for RNA Sequence Design**

This repository contains the inference code for Struct2SeQ, a deep reinforcement learning framework that generates RNA sequences conditioned on target secondary structures and SHAPE reactivity constraints. The model uses deep Q-learning to design sequences that fold into desired structures, including complex pseudoknots.

## Paper

**Struct2SeQ: RNA Inverse Folding with Deep Q-Learning**  
Shujun He and Qing Sun  
*Preprint, January 2026*

## Overview

Struct2SeQ formulates RNA sequence design as a sequential decision-making process, using a transformer-based encoder-decoder architecture trained with deep Q-learning. The model learns to construct sequences that:
- Fold into intended secondary structures (including pseudoknots)
- Exhibit experimentally consistent SHAPE reactivity profiles
- Explore diverse sequence space while maintaining structural accuracy

In the OpenKnot challenges, Struct2SeQ achieved human-level performance on pseudoknot design tasks while generating substantially more diverse solutions than human players.

## Key Features

- **Deep Q-Learning Framework**: Uses reinforcement learning to learn structure-conditioned sequence generation
- **Multiple Sampling Strategies**: 
  - Greedy sampling with varying exploration (p=0.05, 0.1)
  - Stochastic sampling for diversity
  - Optional beam search (top-k)
- **Wildtype Sequence Bias**: Optionally biases generation toward reference sequences
- **Rescue Mechanism**: Targeted mutation strategy for near-perfect sequences
- **SHAPE Profile Integration**: Generates and evaluates SHAPE reactivity profiles
- **Pseudoknot Support**: Handles complex RNA structures including pseudoknots
- **Batch Processing**: Efficient processing of multiple target structures

## Performance

- **OpenKnot Round 7 (100mer)**: 19/20 puzzles solved, human-competitive performance
- **OpenKnot Round 7b (240mer)**: Significantly outperformed human players (p=0.024)
- **Sequence Diversity**: Generates solutions averaging >50 mutations from wildtype
- **Throughput**: Orders of magnitude more unique solutions than human designers

## Requirements

### Core Dependencies
```
torch
polars
numpy
matplotlib
tqdm
```

### Custom Modules (included in repository)
- `Functions` - Utility functions for sequence/structure manipulation
- `Dataset` - Data loading and preprocessing
- `Encoder_Decoder` - Transformer model architecture (DotBracketRNATransformer)
- `search_v2` - Sequence generation and search algorithms
- `Env` - Environment wrapper (DQN_env) with RibonanzaNet integration

### Pre-trained Model
- `policy_network.pt` - Trained Q-network weights (required for inference)

## Installation

1. Clone the repository
2. Install dependencies: `pip install torch polars numpy matplotlib tqdm`
3. Ensure the trained model `policy_network.pt` is in the working directory
4. Verify all custom modules are available

## Usage

### Basic Command

```bash
python inference.py \
    --target_df input_structures.csv \
    --output_csv results.csv \
    --out_folder output_results \
    --gpu_id 0 \
    --n_structures 100
```

### Required Arguments

- `--target_df`: Path to input CSV file containing target structures
  - Must have columns: `Title`, `Dot-bracket`, `wild_type_sequence`
- `--output_csv`: Filename for output CSV (default: `output.csv`)

### Optional Arguments

- `--gpu_id`: GPU device ID (default: `0`)
- `--n_structures`: Number of structures to process (default: `100`)
- `--out_folder`: Output directory for results (default: `results`)

## Input Format

The input CSV file must contain the following columns:

| Column | Description | Format |
|--------|-------------|--------|
| `Title` | Unique identifier for each structure | String |
| `Dot-bracket` | Target secondary structure | Dot-bracket notation (e.g., `(((...)))`) |
| `wild_type_sequence` | Reference RNA sequence (optional) | ACGU nucleotides |

### Dot-Bracket Notation
- `(` and `)` - Base pairs
- `.` - Unpaired nucleotides
- Pseudoknots supported via extended notation

### Example Input
```csv
Title,Dot-bracket,wild_type_sequence
Pseudoknot_1,(((...[[[)))...]]]...,GCCCAAAUUUGGGAAACCCAAA
Hairpin_1,((((....)))),GGGGAAAACCCC
```

## Output Format

The output CSV contains comprehensive results for all generated sequences:

| Column | Description |
|--------|-------------|
| `sequence` | Designed RNA sequence (ACGU) |
| `predicted_structure` | Predicted secondary structure (dot-bracket) |
| `source` | Structure identifier from input (Title) |
| `shape_profile` | Predicted SHAPE reactivity profile (list of floats) |
| `target_structure` | Target secondary structure from input |
| `jaccard` | Jaccard similarity between predicted and target base pairs (0-1) |
| `structure_match` | Boolean indicating perfect structural match |

### Evaluation Metrics

**Jaccard Similarity**: Intersection over union of predicted and target base-pair sets
- 1.0 = Perfect match
- 0.0 = No overlap

**SHAPE Profile**: Predicted reactivity values (0-2+)
- Higher values indicate greater nucleotide flexibility/accessibility
- Typically unpaired positions have SHAPE > 0.5
- Typically paired positions have SHAPE < 0.25

## Algorithm Details

### Sequence Generation Pipeline

1. **Initial Sampling** (default: 128 sequences per structure):
   - **Greedy sampling** with p=0.05 (low exploration)
   - **Greedy sampling** with p=0.1 (moderate exploration)
   - **Stochastic sampling** with p=1.0 (high diversity)
   - **Optional beam search** (GPU 0 only, if enabled)

2. **Structure Prediction**: 
   - Each generated sequence is folded using the environment's structure predictor (RibonanzaNet-SS)
   - SHAPE reactivity profiles computed using RibonanzaNet
   - Base pairs extracted via Hungarian matching algorithm

3. **Evaluation**:
   - Calculate Jaccard similarity between predicted and target base pairs
   - Compute OpenKnot score (if SHAPE constraints enabled)

4. **Rescue Mechanism** (applied if Jaccard < 1.0):
   - Identify positions where predicted structure differs from target
   - If ≤4 differing positions: enumerate all valid nucleotide combinations
   - Re-evaluate mutated sequences
   - Select best-performing sequence

5. **Selection**: Choose sequence with highest Jaccard similarity

### Deep Q-Learning Framework

The model was trained using a variant of Deep Q-Learning with:
- **Twice-shifted action value training** for autoregressive generation
- **Position-dependent reward** based on structural correctness and SHAPE compliance
- **Exponential reward reweighting** to encourage near-perfect → perfect transitions
- **Linearly decayed discount factor** (γ_t) to emphasize early position decisions

### Wildtype Sequence Bias

When `wild_type_sequence` is provided:
- Random upweighting bias sampled uniformly between 0.75-1.0
- Balances similarity to wildtype with structural accuracy
- Applied through weighted sampling during sequence generation

## Performance Metrics

The tool reports:
- Average Jaccard similarity across all designs
- Number of perfectly solved structures (Jaccard = 1.0)
- Hamming distance from wildtype sequence
- Total runtime

## Model Architecture

Struct2SeQ uses a sequence-to-sequence transformer architecture that combines convolutional, transformer, and graph-based modules:

### Encoder (Dot-Bracket Structure → Representation)
- **Embedding Layer**: Dot-bracket tokens → continuous embeddings
- **LSTM Positional Encoding**: Learned positional information
- **1D Convolution**: Local structural motif extraction
- **Graph Convolution Network (GCN)**: Integrates pairwise contact information for long-range base-pairing
- **Transformer Encoder** (6 layers): Global context aggregation via multi-head self-attention

### Decoder (Structure Representation → Q-values)
- **Token Embedding**: Partial sequence → continuous embeddings
- **Paired/Unpaired Encoding**: Binary structural conditioning
- **LSTM Positional Encoding**: Learned positional information
- **Causal 1D Convolution**: Ensures autoregressive decoding
- **Causal Transformer Decoder** (6 layers): 
  - Masked self-attention
  - Encoder-decoder cross-attention
  - Position-wise feedforward layers
  - Key-value caching for efficient inference
- **Output Projection**: Hidden states → Q-values over {A, U, G, C}

### Architecture Specifications
- **Embedding Size**: 384
- **Attention Heads**: 16
- **Encoder Layers**: 6
- **Decoder Layers**: 6
- **Feedforward Dimension**: 1536 (4× embedding size)
- **Dropout**: 0.1
- **Vocabulary**: Dot-bracket (100 tokens), RNA (5 tokens including padding)

## Benchmark Results

### OpenKnot Round 7 (100mer Pseudoknots)
- **Puzzles Solved**: 19/20 (95%) - matching human performance
- **Unique Solutions Generated**: Orders of magnitude more than human players
- **Average Mutation Distance**: >50 nucleotides from wildtype (vs. small distances for human designs)
- **Training**: 120 A6000 GPU-hours (15 hrs on 8×A6000)
- **Inference Budget**: 98,000 sequences per target structure

### OpenKnot Round 7b (240mer Pseudoknots)
- **Struct2SeQ-SHAPE Performance**: 
  - Significantly outperformed human players (p=0.024, Wilcoxon test)
  - 2× summed Z-scores compared to Eterna players
- **Struct2SeQ Performance**: Human-competitive (p=0.84)
- **Training**: 4,000 A100 GPU-hours (15 hrs on 32 nodes, 256 GPUs)
- **Training Data**: Human genome scanned in 240mer windows (stride=50)

### Key Findings
- **Struct2SeQ-SHAPE** generates sequences with better OpenKnot scores (experimental SHAPE compliance)
- **SHAPE constraints** are critical for experimental performance
- Successfully scales to complex, long-range pseudoknotted structures
- Explores substantially larger sequence space than human-designed solutions

## Tips for Best Results

1. **GPU Memory**: Batch size automatically set to 32 for structure prediction
2. **Rescue Mechanism**: Most effective for near-matches (≤4 differing positions)
3. **Sampling Diversity**: Uses 3-4 different sampling strategies for robust coverage
4. **Wildtype Bias**: Adjust `up_bias` range (0.75-1.0) for different conservation levels

## Troubleshooting

- **CUDA Out of Memory**: Reduce batch size in structure prediction sections
- **Missing Modules**: Ensure all custom modules are in Python path
- **Model Not Found**: Verify `policy_network.pt` exists in working directory
- **Invalid Structures**: Check dot-bracket notation is properly formatted

## Citation

If you use Struct2SeQ in your research, please cite:

```bibtex
@article{he2026struct2seq,
  title={Struct2SeQ: RNA Inverse Folding with Deep Q-Learning},
  author={He, Shujun and Sun, Qing},
  journal={Preprint},
  year={2026},
  month={January}
}
```

## Acknowledgements

We thank Rhiju Das and the Eterna team for organizing the OpenKnot challenges, and Jill Towney and Jonathan Romano for providing valuable target structures for training of 100mer Struct2SeQ models.

## Funding

This work was supported by:
- National Institutes of Health (R01AI165433)
- Texas A&M University X-grants
- NAIRR pilot program

## Related Work

- **RibonanzaNet**: Deep learning framework for RNA structure and SHAPE prediction ([He et al. 2024](https://doi.org/10.1101/2024.02.24.581671))
- **OpenKnot Challenges**: Community-driven RNA design competitions organized by Eterna

## License

[Specify license here]

## Authors

**Shujun He**  
Artie McFerrin Department of Chemical Engineering  
Texas A&M University  
Email: shujun@tamu.edu

**Qing Sun**  
Artie McFerrin Department of Chemical Engineering  
Texas A&M University  
Email: sunqing@tamu.edu

## Issues and Support

For questions, bug reports, or feature requests, please open an issue on the GitHub repository.

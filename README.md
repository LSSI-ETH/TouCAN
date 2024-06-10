# TouCAN: a tool for TCR clustering by contrastive learning on antigen specificity 

## Abstract
Effective clustering of T-cell receptor (TCR) sequences could be used to predict their antigen-specificities. TCRs with highly dissimilar sequences can bind to the same antigen, thus making their clustering into a common antigen group a central challenge. Here, we develop TouCAN, a method that relies on contrastive learning and pre-trained protein language models to perform TCR sequence clustering and antigen-specificity predictions.  Following training, TouCAN  demonstrates the ability to cluster highly dissimilar TCRs into common antigen groups. Additionally, TouCAN demonstrates TCR clustering performance and antigen-specificity predictions comparable to other leading methods in the field.

## Dependencies
Dependencies: python(version>3.0.0) ; tensorflow (version>1.5.0) ; numpy (version=1.16.3) ; keras (version=2.2.4) ; pandas (version=0.23.4) ; scikit-learn (version=0.20.3) ; scipy (version=1.2.1)

## Usage and Data Formatting
Your data should be in the following format to use TouCAN:
 - tcr_data.csv file containing CDR3a, Va, CDR3b, Vb and antigen amino acid sequences

The model can be easily re-trained on a new TCR-antigen dataset using the one-hot encoding of CDR loops. However, TouCAN performs best when trained on ESM-1v encodings of V-domains of TCRɑ and TCRβ chains that need to be obtained independently. Please, follow the instructions of the ESM GitHub page on how to obtain encodings here: https://github.com/facebookresearch/esm . To obtain TCR V-domain sequence from the known CDR3, V and J gene information, you can use Stitchr: https://jamieheather.github.io/stitchr/

## You can train your own model or simply predict TCR epitope clustering and classification on your data:
#### To train TouCAN on your own data: 
##### TCR input parameters:
 - input_type : "beta_chain or paired_chain"
 - encoding_type: "onehot or ESM"
 - esm_type: "ESM1v or ESM2"
 - chain_type: "TCR or ab_VC or ab_V"

##### Model parameters
 - embedding_space: embedding dimension for TCR (integer)
 - epochs: N of epochs to train the model (integer)
 - patience: early stopping with patience for N epochs (integer)
 - batch_size: specify the batch size
 - learning_rate: specify the learning rate
 - triplet_loss: "hard or semihard"
    
##### Visualize TouCAN embeddings 
 - plot_embedding: True or False

Example command:

python TouCAN_train.py --input_type paired_chain --encoding_type ESM --esm_type ESM1v --chain_type ab_V --embedding_space 160 --epochs 200 --patience 15 --batch_size 256 --learning_rate 0.0005


#### To predict TCR epitope labels and clustering: 
##### TCR input parameters:
 - input_type : "beta_chain or paired_chain"
 - encoding_type: "onehot or ESM"
 - esm_type: "ESM1v or ESM2"
 - chain_type: "TCR or ab_VC or ab_V"

Example command:

python TouCAN_predict.py --input_type paired_chain --encoding_type ESM --esm_type ESM1v --chain_type ab_V --output output_file

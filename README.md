# TouCAN: a tool for TCR clustering by contrastive learning on antigen specificity 

## Abstract
Effective clustering of T-cell receptor (TCR) sequences could be used to predict their antigen-specificities. TCRs with highly dissimilar sequences can bind to the same antigen, thus making their clustering into a common antigen group a central challenge. Here, we develop TouCAN, a method that relies on contrastive learning and pre-trained protein language models to perform TCR sequence clustering and antigen-specificity predictions.  Following training, TouCAN  demonstrates the ability to cluster highly dissimilar TCRs into common antigen groups. Additionally, TouCAN demonstrates TCR clustering performance and antigen-specificity predictions comparable to other leading methods in the field.

## Dependencies
Dependencies: python(version>3.0.0) ; tensorflow (version>1.5.0) ; numpy (version=1.16.3) ; keras (version=2.2.4) ; pandas (version=0.23.4) ; scikit-learn (version=0.20.3) ; scipy (version=1.2.1)

## Usage and Data Formatting
Your data should be in the following format to use TouCAN:
 - tcr_data.csv file containing CDR3a, Va, CDR3b, Vb and antigen amino acid sequences

The model can be easily re-trained on a new TCR-antigen dataset using the one-hot encoding of CDR loops. However, TouCAN performs best when trained on ESM-1v encodings of V-domains of TCRɑ and TCRβ chains that need to be obtained independently. Please, follow the instructions of the ESM GitHub page on how to obtain encodings here: https://github.com/facebookresearch/esm . To obtain TCR V-domain sequence from the known CDR3, V and J gene information, you can use Stitchr: https://jamieheather.github.io/stitchr/

## Example file is put under the example/example_df.csv
Command: python TouCAN.py -input example/example_df.csv -output example/output

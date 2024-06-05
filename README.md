# CVAE-Tacotron
Code for the CVAE-NL model from the Accented Text-to-Speech Synthesis with a Conditional Variational Autoencoder paper.

Paper available at: https://arxiv.org/abs/2211.03316

Sample site available at: https://dapwner.github.io/CVAE-Tacotron/

![alt text](https://github.com/Dapwner/CVAE-Tacotron/blob/main/schematic.png)

## Training
First, download your dataset and preprocess the audio data into mel spectrogram .npy arrays with the preprocess.py script
Then, to train the model, run CUDA_VISIBLE_DEVICES=X python train.py --dataset L2Arctic

## Inference
Once trained, you can run CUDA_VISIBLE_DEVICES=X python synthesize.py --dataset L2Arctic --restore_step [N] --mode [batch/single/sample] --text [TXT] --speaker_id [SPID] --accent [ACC]

Or run synthesize_debug.py in a debugger and figure out!

###Inference modes

"single": takes in a reference audio to be used for both speaker and accent branches

"batch": allows to extract mu and std values for speakers and accents from a passed dataset

**"sample":** take the extracted mu and std values, choose your speaker with --speaker_id, and your accent with --accent, then synthesize the speech!

## BibTeX
'''
@article{melechovsky2022accented,
  title={Accented Text-to-Speech Synthesis with a Conditional Variational Autoencoder},
  author={Melechovsky, Jan and Mehrish, Ambuj and Sisman, Berrak and Herremans, Dorien},
  journal={arXiv preprint arXiv:2211.03316},
  year={2022}
}
'''

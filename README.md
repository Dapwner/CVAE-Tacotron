# CVAE-Tacotron
Code for the CVAE-NL model from the Accented Text-to-Speech Synthesis with a Conditional Variational Autoencoder paper, available at: https://arxiv.org/abs/2211.03316
Sample site available at: https://dapwner.github.io/CVAE-Tacotron/

![alt text](https://github.com/Dapwner/CVAE-Tacotron/blob/main/schematic.png?raw=true)

## Training
First preprocess your data into mel spectrogram .npy arrays with the preprocess.py script
Then run CUDA_VISIBLE_DEVICES=X python train.py --dataset L2Arctic

## Inference
Once trained, you can run CUDA_VISIBLE_DEVICES=X python synthesize.py --dataset L2Arctic --restore_step [N] --mode [batch/single/sample] --text [TXT] --speaker_id [SPID] --accent [ACC]

Or run synthesize_debug.py in a debugger and figure out!

###Inference modes

"single": takes in a reference audio to be used for both speaker and accent branches

"batch": allows to extract mu and std values for speakers and accents from a passed dataset

**"sample":** take the extracted mu and std values, choose your speaker with --speaker_id, and your accent with --accent, then synthesize the speech!

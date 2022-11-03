import os
import json
# import re
import argparse
# from string import punctuation

import torch
import numpy as np
from torch.utils.data import DataLoader
# from g2p_en import G2p
# from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, to_device, infer_one_sample, plot_embedding #, read_lexicon
from dataset import TextDataset, Dataset
from text import text_to_sequence, sequence_to_text
from utils.tools import pad_2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_english(text, preprocess_config):
    sequence = text_to_sequence(
        text, preprocess_config["preprocessing"]["text"]["text_cleaners"]
    )
    print("Raw Text Sequence: {}".format(text))
    print("Sequence: {}".format(" ".join([str(id) for id in sequence_to_text(sequence)])))
    print("Sequence Input: {}".format(" ".join([str(id) for id in sequence])))
    return np.array(sequence)

# use this one to synthesize using extracted stats = stored mu (and sigma) values
def synthesize_sample(model, args, configs, mel_stats, vocoder, batchs):
    preprocess_config, model_config, train_config = configs
    n_frames_per_step = model_config["decoder"]["n_frames_per_step"]
    arraypath = train_config["path"]["array_path"]
    for batch in batchs:

        # max_target_len = batch[4].shape[1]
        # r_len_pad = max_target_len % n_frames_per_step
        # if r_len_pad != 0:
        #     max_target_len += n_frames_per_step - r_len_pad
        #     assert max_target_len % n_frames_per_step == 0
            
        # ss = pad_2D(batch[4], max_target_len)
        batch=to_device(batch, device, mel_stats)
        # batch=to_device((*batch[0:4], ss, *batch[5:]), device, mel_stats)
        # accents2 = torch.LongTensor(accents2).to(device)

        # if flat_acc:
        #     std_acc = sig_acc*torch.ones((1,model_config["accent_encoder"]["z_dim"])).to(device)
        # else:
        #     std_acc = sig_acc*torch.randn((1,model_config["accent_encoder"]["z_dim"])).to(device)

        # if flat_spk:           
        #     std_spk = sig_spk*torch.ones((1,model_config["speaker_encoder"]["z_dim"])).to(device)
        # else:
        #     std_spk = sig_spk*torch.randn((1,model_config["speaker_encoder"]["z_dim"])).to(device)
        
        acc_mu=np.load(os.path.join(arraypath,'acc_mu.npy'))
        acc_var=np.load(os.path.join(arraypath,'acc_var.npy'))

        spk_mu=np.load(os.path.join(arraypath,'spk_mu.npy'))
        spk_var=np.load(os.path.join(arraypath,'spk_var.npy'))

        acc_id=np.load(os.path.join(arraypath,'acc_id.npy'))
        spk_id=np.load(os.path.join(arraypath,'spk_id.npy'))
        
        z_acc=np.mean(acc_mu[acc_id==batch[8][0].cpu().item()],axis=0)
        z_spk=np.mean(spk_mu[spk_id==batch[2][0].cpu().item()],axis=0)
        accents2=None

        z_acc=torch.from_numpy(z_acc).unsqueeze(0).to(device)
        z_spk=torch.from_numpy(z_spk).unsqueeze(0).to(device)


        with torch.no_grad():
            #forward
            # output = model(*batch[2:4], batch[5], batch[5], batch[4], batch[4].size(1), accents=batch[-1])
            # output = model(*batch[2:4], batch[5], batch[5], batch[4], torch.tensor(max_target_len).reshape(-1).to(device), accents=batch[-1])
            model.eval()
            output = model.inference_sampling(*batch[2:5], *batch[6:9], accents2, z_acc, z_spk, args)
            infer_one_sample(
                    batch,
                    output,
                    vocoder,
                    mel_stats,
                    model_config,
                    preprocess_config,
                    train_config["path"]["result_path"],
                    args,
                        )



# use this to synthesize with a reference audio
def synthesize_single(model, args, configs, mel_stats, vocoder, batchs):
    preprocess_config, model_config, train_config = configs
    n_frames_per_step = model_config["decoder"]["n_frames_per_step"]
    
    for batch in batchs:
        # max_target_len = batch[4].shape[1]
        # r_len_pad = max_target_len % n_frames_per_step
        # if r_len_pad != 0:
        #     max_target_len += n_frames_per_step - r_len_pad
        #     assert max_target_len % n_frames_per_step == 0
            
        # ss = pad_2D(batch[4], max_target_len)
        batch=to_device(batch, device, mel_stats)
        # batch=to_device((*batch[0:4], ss, *batch[5:]), device, mel_stats)
        with torch.no_grad():
            #forward
            # output = model(*batch[2:4], batch[5], batch[5], batch[4], batch[4].size(1), accents=batch[-1])
            # output = model(*batch[2:4], batch[5], batch[5], batch[4], torch.tensor(max_target_len).reshape(-1).to(device), accents=batch[-1])
            model.eval()
            output = model.inference(*batch[2:5], *batch[6:9])
            infer_one_sample(
                    batch,
                    output,
                    vocoder,
                    mel_stats,
                    model_config,
                    preprocess_config,
                    train_config["path"]["result_path"],
                    args,
                    )
            

# use this to run a whole set of reference audios through the model and extract mu and sigma values for it (to be used for inference without ref audio)
def synthesize_batch(model, args, configs, mel_stats, vocoder, loader):
    preprocess_config, model_config, train_config = configs
    normalize = preprocess_config["preprocessing"]["mel"]["normalize"]
    acc_mu = []
    acc_var = []
    spk_mu = []
    spk_var = []

    embedding_accent_id = []
    embedding_speaker_id = []

    i=0

    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device, mel_stats if normalize else None)
            with torch.no_grad():
                ids,raw_texts, speakers, texts, text_lens,max_text_lens, mels,mel_lens,max_target_len,r_len_pad,gates,spker_embeds,accents = batch
                batch  = (ids, raw_texts, speakers, texts, mels,text_lens, max_text_lens, spker_embeds, accents)
                #forward
                model.eval()
                output = model.inference(*batch[2:5], *batch[6:9])
                infer_one_sample(
                    batch,
                    output,
                    vocoder,
                    mel_stats,
                    model_config,
                    preprocess_config,
                    train_config["path"]["result_path"],
                    args,
                )
                prob_ = output[4]
                acc_mu.append(prob_[0].squeeze(0).cpu().detach())
                acc_var.append(prob_[1].squeeze(0).cpu().detach())

                spk_mu.append(prob_[2].squeeze(0).cpu().detach())
                spk_var.append(prob_[3].squeeze(0).cpu().detach())

                embedding_accent_id.append(batch[8].cpu().detach())
                embedding_speaker_id.append(batch[2].cpu().detach())

                print(i)
                i+=1
            
    
    embedding_acc = np.array([np.array(xi) for xi in acc_mu])
    embedding_acc_var = np.array([np.array(xi) for xi in acc_var])
    embedding_accent_id = np.array([np.array(id_[0]) for id_ in embedding_accent_id])

    embedding_spk = np.array([np.array(xi) for xi in spk_mu])
    embedding_spk_var = np.array([np.array(xi) for xi in spk_var])
    embedding_speaker_id = np.array([np.array(id_[0]) for id_ in embedding_speaker_id])

    np.save('output/arrays/acc_mu.npy',embedding_acc)
    np.save('output/arrays/acc_var.npy',embedding_acc_var)
    np.save('output/arrays/spk_mu.npy',embedding_spk)
    np.save('output/arrays/spk_var.npy',embedding_spk_var)
    np.save('output/arrays/acc_id.npy',embedding_accent_id)
    np.save('output/arrays/spk_id.npy',embedding_speaker_id)






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single", "sample"],
        required=True,
        default="sample"
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default='He turned sharply and faced Gregson across the table',
        help="raw text to synthesize, for single and sample modes only",
    )
    parser.add_argument(
        "--speaker_id",
        type=str,
        default="ABA",
        help="speaker ID for sample mode only, for single mode this should be the same as the reference audio speaker",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="ABA_a0009",
        help="Reference audio for the speaker, for single-sentence mode only",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="L2Arctic",
        help="name of dataset",
    )
    parser.add_argument(
        "--accent",
        type=str,
        default="Arabic",
        help="Accent name, e.g., Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese",
    )
    args = parser.parse_args()


    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)
    with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        mel_stats = stats["mel"]
    os.makedirs(
        os.path.join(train_config["path"]["result_path"], str(args.restore_step)), exist_ok=True)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
#        dataset = TextDataset(args.source, preprocess_config, model_config)
#        batchs = DataLoader(
#            dataset,
#            batch_size=1, # currently only 1 is supported
#            collate_fn=dataset.collate_fn,
#        )
        dataset = Dataset(args.source, preprocess_config, model_config, train_config, sort=True, drop_last=True)
        batch_size = train_config["optimizer"]["batch_size"]
        group_size = 4  # Set this larger than 1 to enable sorting in Dataset
        assert batch_size * group_size < len(dataset)
        loader = DataLoader(
                dataset,
                batch_size=batch_size*group_size,
                shuffle=False,
                collate_fn=dataset.collate_fn,
                )

        synthesize_batch(model, args, configs, mel_stats, vocoder, loader)
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]

        # Speaker Info
        load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            speaker_map = json.load(f)
        speakers = np.array([speaker_map[args.speaker_id]]) if model_config["multi_speaker"] else np.array([0]) # single speaker is allocated 0
        spker_embed = np.load(os.path.join(
            preprocess_config["path"]["preprocessed_path"],
            "spker_embed",
            "{}-spker_embed.npy".format(args.speaker_id),
        )) if load_spker_embed else None

        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        else:
            raise NotImplementedError
        ref_spk, ref_sample = args.basename.split("_")
        text_lens = np.array([len(texts[0])])
        mel_path = os.path.join(
                preprocess_config["path"]["preprocessed_path"],
                "mel",
            "{}-mel-arctic_{}.npy".format(ref_spk, ref_sample),
        )
        mel = np.load(mel_path)
        mel = np.expand_dims(mel,axis=0)
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "accents.json")) as f:
            accent_map = json.load(f)
            
        accents_to_indices = dict()
        
        for _idx, acc in enumerate(preprocess_config['accents']):
            accents_to_indices[acc] = _idx
      
        accents = np.array([accents_to_indices[accent_map[ref_spk]]])
        loader = [(ids, raw_texts, speakers, texts, mel,text_lens, max(text_lens), spker_embed,accents)]
        
        synthesize_single(model, args, configs, mel_stats, vocoder, loader)



    if args.mode == "sample":
        ids = raw_texts = [args.text[:100]]
        # sig_acc = args.siga
        # sig_spk = args.sigs
        # flat_acc = args.flata
        # flat_spk = args.flats

        # Speaker Info
        load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            speaker_map = json.load(f)
        speakers = np.array([speaker_map[args.speaker_id]]) if model_config["multi_speaker"] else np.array([0]) # single speaker is allocated 0
        spker_embed = np.load(os.path.join(
            preprocess_config["path"]["preprocessed_path"],
            "spker_embed",
            "{}-spker_embed.npy".format(args.speaker_id),
        )) if load_spker_embed else None

        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        else:
            raise NotImplementedError
        acc_name=args.accent
        acc_name2=args.accent2

        text_lens = np.array([len(texts[0])])


        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "accents.json")) as f:
            accent_map = json.load(f)
            
        accents_to_indices = dict()
        
        for _idx, acc in enumerate(preprocess_config['accents']):
            accents_to_indices[acc] = _idx
        
        mel=np.zeros((1,1,1))
        accents = np.array([accents_to_indices[acc_name]])
        accents2 = np.array([accents_to_indices[acc_name2]])
        loader = [(ids, raw_texts, speakers, texts, mel, text_lens, max(text_lens), spker_embed, accents)]
        
        synthesize_sample(model, args, configs, mel_stats, vocoder, loader)



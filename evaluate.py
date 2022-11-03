import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from model import Tacotron2Loss
from dataset import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, mel_stats, logger=None, vocoder=None, len_losses=3):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, model_config, train_config, sort=True, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    normalize = preprocess_config["preprocessing"]["mel"]["normalize"]

    # Get loss function
    Loss = Tacotron2Loss(preprocess_config, model_config, train_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(len_losses)]
    # inf_loss_sums = [0 for _ in range(len_losses)]

    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device, mel_stats if normalize else None)
            with torch.no_grad():
                # Forward
                output = model(*(batch[2:]))

                losses = Loss(batch, output, step)

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

                # Inference

                # output_inference = model.inference(*batch[2:5], *batch[6:9])





                # Cal Loss
                # inf_loss = Loss(batch, output_inference, step)



                # for i in range(len(inf_losses)):
                #     inf_loss_sums[i] += inf_losses[i].item() * len(batch[0])

    # s1=batch[2]
    # s2=batch[3]
    output_inference = model.module.inference(batch[2][0].unsqueeze(0), batch[3][0].unsqueeze(0), batch[6][0].unsqueeze(0), batch[5], batch[11], batch[12][0].unsqueeze(0))

    # output_inference = model.module.inference(*batch[2:4], batch[6], batch[5], *batch[11:])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]
    # inf_loss_means = [loss_sum / len(dataset) for loss_sum in inf_loss_sums]


    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Gate Loss: {:.4f}, Guided Attention Loss: {:.4f}, Encoder Loss: {:.4f} ".format(
        *([step] + [l for l in loss_means])
    )
    # message = "Inference at Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Gate Loss: {:.4f}, Guided Attention Loss: {:.4f}, Encoder Loss: {:.4f} ".format(
    #     *([step] + [l for l in inf_loss_means])
    # )
    if logger is not None:
        fig, gate_fig, wav_reconstruction, wav_prediction, wav_inference, tag = synth_one_sample(
            batch,
            output,
            output_inference,
            vocoder,
            mel_stats,
            model_config,
            preprocess_config,
            step,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        log(
            logger,
            step=step,
            fig=gate_fig,
            tag="Gates/validation",
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )
        log(
            logger,
            audio=wav_inference,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_inferred".format(step, tag),
        )

    return message

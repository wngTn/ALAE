# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

from net import *
from model import Model
from launcher import run
from dataloader import *
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
from PIL import Image
import PIL
import cv2


def draw_uncurated_result_figure(cfg, png, model, cx, cy, cw, ch, rows, lods, seed):
    rnd = np.random.RandomState()
    latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE)
    samplez = torch.tensor(latents).float().cuda()
    image = model.generate(cfg.DATASET.MAX_RESOLUTION_LEVEL-2, 1, samplez, 1, mixing=True)[0]
    im = image.cpu().numpy()
    im = im.transpose(1, 2, 0)
    im = im * 0.5 + 0.5
    im = np.clip(im * 255, 0, 255).astype(np.uint8)
    bgr_image = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output/test.jpg', bgr_image)
    


def sample(cfg, logger):
    torch.cuda.set_device(0)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        style_mixing_prob=cfg.MODEL.STYLE_MIXING_PROB,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER)

    model.cuda(0)
    model.eval()
    model.requires_grad_(False)

    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_d
    mapping_fl = model.mapping_f

    dlatent_avg = model.dlatent_avg

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    model_dict = {
        'discriminator_s': encoder,
        'generator_s': decoder,
        'mapping_tl_s': mapping_tl,
        'mapping_fl_s': mapping_fl,
        'dlatent_avg': dlatent_avg
    }

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {},
                                logger=logger,
                                save=False)

    checkpointer.load()

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    decoder = nn.DataParallel(decoder)

    im_size = 2 ** (cfg.MODEL.LAYER_COUNT + 1)
    with torch.no_grad():
        draw_uncurated_result_figure(cfg, 'make_figures/output/%s/generations.jpg' % cfg.NAME,
                                     model, cx=0, cy=0, cw=im_size, ch=im_size, rows=6, lods=[0, 0, 0, 1, 1, 2], seed=5)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='ALAE-generations', default_config='configs/ffhq.yaml',
        world_size=gpu_count, write_log=False)

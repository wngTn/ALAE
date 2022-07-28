# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch.utils.data
from torchvision.utils import save_image
from net import *
from model import Model
from launcher import run
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import lreq
from PIL import Image
import random
from glob import glob


lreq.use_implicit_lreq.set(True)


def sample(cfg, logger):
    torch.cuda.set_device(0)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
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

    extra_checkpoint_data = checkpointer.load()

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    def encode(x):
        Z, _ = model.encode(x, layer_count - 1, 1)
        Z = Z.repeat(1, model.mapping_f.num_layers, 1)
        return Z

    def decode(x):
        layer_idx = torch.arange(2 * cfg.MODEL.LAYER_COUNT)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)
        # x = torch.lerp(model.dlatent_avg.buff.data, x, coefs)
        return model.decoder(x, layer_count - 1, 1, noise=True)

    im_size = 2 ** (cfg.MODEL.LAYER_COUNT + 1)

    def do_attribute_traversal(x, attrib_idx, value):
        # img = np.asarray(Image.open(path))

        factor = x.shape[2] // im_size
        if factor != 1:
            x = torch.nn.functional.avg_pool2d(x[None, ...], factor, factor)[0]
        assert x.shape[2] == im_size
        _latents = encode(x[None, ...].cuda())
        latents = _latents[0, 0]

        latents -= model.dlatent_avg.buff.data[0]

        w0 = torch.tensor(np.load("principal_directions/direction_%d.npy" % attrib_idx), dtype=torch.float32)

        attr0 = (latents * w0).sum()

        latents = latents - attr0 * w0

        def update_image(w):
            with torch.no_grad():
                w = w + model.dlatent_avg.buff.data[0]
                w = w[None, None, ...].repeat(1, model.mapping_f.num_layers, 1)

                layer_idx = torch.arange(model.mapping_f.num_layers)[np.newaxis, :, np.newaxis]
                cur_layers = (7 + 1) * 2
                mixing_cutoff = cur_layers
                styles = torch.where(layer_idx < mixing_cutoff, w, _latents[0])

                x_rec = decode(styles)
                return x_rec
    
        W = latents + w0 * (attr0 + value)
        im = update_image(W)[0]

        return im

        

    cfg_labels = {
        0 : "GENDER",
        1 : "SMILE",
        2 : "ATTRACTIVE",
        3 : "WAVY_HAIR",
        4 : "YOUNG",
        10 : "BIG_LIPS",
        11 : "BIG_NOSE",
        17 : "CHUBBY",
        19 : "GLASSES"
    }
    
    template_images = list(sorted(glob(cfg.CUSTOM.IMAGE_PATH + '/*.png')))
    for i, template_image in enumerate(template_images):
        img = np.asarray(Image.open(template_image))
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = img.transpose((2, 0, 1))
        x = torch.tensor(np.asarray(img, dtype=np.float32), device='cpu', requires_grad=True).cuda() / 127.5 - 1.
        if x.shape[0] == 4:
            x = x[:3]

        num_changes = 0
        for attr_id, label in cfg_labels.items():
            attr_value = eval(f"cfg.CUSTOM.{label}[{i}]") if len(eval(f"cfg.CUSTOM.{label}")) > i else None
            if attr_value is not None:
                num_changes += 1
                x = do_attribute_traversal(x, attr_id, attr_value)
        if num_changes == 0:
            x = do_attribute_traversal(x, 1, 0)
        save_image(x * 0.5 + 0.5, "data/face_gen/faces/face_gen_from_%s" % os.path.basename(template_image))



if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='ALAE-traversals', default_config='configs/ffhq.yaml',
        world_size=gpu_count, write_log=False)

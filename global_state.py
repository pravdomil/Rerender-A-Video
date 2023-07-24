import os
from enum import Enum

import huggingface_hub
import torch
from safetensors.torch import load_file

import src.import_util  # noqa: F401
from ControlNet.annotator.canny import CannyDetector
from ControlNet.annotator.hed import HEDdetector
from ControlNet.annotator.midas import MidasDetector
from ControlNet.cldm.model import create_model, load_state_dict
from gmflow_module.gmflow.gmflow import GMFlow
from sd_model_cfg import model_dict
from src.controller import AttentionControl
from src.ddim_v_hacked import DDIMVSampler

REPO_NAME = 'Anonymous-sub/Rerender'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ProcessingState(Enum):
    NULL = 0
    FIRST_IMG = 1
    KEY_IMGS = 2


class GlobalState:

    def __init__(self):
        self.sd_model = None
        self.ddim_v_sampler = None
        self.detector_type = None
        self.detector = None
        self.controller = None
        self.processing_state = ProcessingState.NULL

        checkpoint = torch.load('models/gmflow_sintel-0c07dcb3.pth', map_location=lambda storage, loc: storage)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

        flow_model = GMFlow(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            attention_type='swin',
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        ).to(device)
        flow_model.load_state_dict(weights, strict=False)
        flow_model.eval()

        self.flow_model = flow_model

    def update_controller(self, inner_strength, mask_period, cross_period, ada_period, warp_period):
        self.controller = AttentionControl(inner_strength, mask_period, cross_period, ada_period, warp_period)

    def update_sd_model(self, sd_model, control_type):
        if sd_model == self.sd_model:
            return
        self.sd_model = sd_model
        model = create_model('./ControlNet/models/cldm_v15.yaml').cpu()
        if control_type == 'HED':
            model.load_state_dict(
                load_state_dict(
                    huggingface_hub.hf_hub_download('lllyasviel/ControlNet', './models/control_sd15_hed.pth'),
                    location=device
                )
            )
        elif control_type == 'canny':
            model.load_state_dict(
                load_state_dict(
                    huggingface_hub.hf_hub_download('lllyasviel/ControlNet', 'models/control_sd15_canny.pth'),
                    location=device
                )
            )
        elif control_type == 'depth':
            model.load_state_dict(
                load_state_dict(
                    huggingface_hub.hf_hub_download('lllyasviel/ControlNet', 'models/control_sd15_depth.pth'),
                    location=device
                )
            )

        model.to(device)
        sd_model_path = model_dict[sd_model]
        if len(sd_model_path) > 0:
            repo_name = REPO_NAME
            # check if sd_model is repo_id/name otherwise use global REPO_NAME
            if sd_model.count('/') == 1:
                repo_name = sd_model

            model_ext = os.path.splitext(sd_model_path)[1]
            downloaded_model = huggingface_hub.hf_hub_download(repo_name, sd_model_path)
            if model_ext == '.safetensors':
                model.load_state_dict(load_file(downloaded_model),
                                      strict=False)
            elif model_ext == '.ckpt' or model_ext == '.pth':
                model.load_state_dict(
                    torch.load(downloaded_model)['state_dict'], strict=False)

        try:
            model.first_stage_model.load_state_dict(torch.load(
                huggingface_hub.hf_hub_download(
                    'stabilityai/sd-vae-ft-mse-original',
                    'vae-ft-mse-840000-ema-pruned.ckpt'))['state_dict'],
                                                    strict=False)
        except Exception:
            print('Warning: We suggest you download the fine-tuned VAE',
                  'otherwise the generation quality will be degraded')

        self.ddim_v_sampler = DDIMVSampler(model)

    def clear_sd_model(self):
        self.sd_model = None
        self.ddim_v_sampler = None
        if device == 'cuda':
            torch.cuda.empty_cache()

    def update_detector(self, control_type, canny_low=100, canny_high=200):
        if self.detector_type == control_type:
            return
        if control_type == 'HED':
            self.detector = HEDdetector()
        elif control_type == 'canny':
            canny_detector = CannyDetector()
            low_threshold = canny_low
            high_threshold = canny_high

            def apply_canny(x):
                return canny_detector(x, low_threshold, high_threshold)

            self.detector = apply_canny

        elif control_type == 'depth':
            midas = MidasDetector()

            def apply_midas(x):
                detected_map, _ = midas(x)
                return detected_map

            self.detector = apply_midas

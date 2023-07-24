import os
import shutil

import PIL.Image
# noinspection PyUnresolvedReferences
import blendmodes.blend
import cv2
import einops
import numpy
# noinspection PyUnresolvedReferences
import pytorch_lightning
import skimage
# noinspection PyUnresolvedReferences
import torch
import torch.nn.functional as functional
import torchvision.transforms as transforms

import ControlNet.annotator.util
import flow.flow_utils
import global_state
import src.config
import src.img_util
import src.import_util  # noqa: F401
import src.video_util

blur = transforms.GaussianBlur(kernel_size=(9, 9), sigma=(18, 18))


def process1():
    cfg = get_config()

    state = global_state.GlobalState()
    state.update_sd_model(cfg.sd_model, cfg.control_type)
    state.update_controller(cfg.inner_strength, cfg.mask_period, cfg.cross_period, cfg.ada_period, cfg.warp_period)
    state.update_detector(cfg.control_type, cfg.canny_low, cfg.canny_high)
    state.processing_state = global_state.ProcessingState.FIRST_IMG

    model = state.ddim_v_sampler.model
    model.control_scales = [cfg.control_strength] * 13
    model.cond_stage_model.device = global_state.device
    model.to(global_state.device)

    input_image = cv2.imread(cfg.input_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = ControlNet.annotator.util.HWC3(input_image)

    return generate_first_img(cfg, state, input_image, 1 - cfg.x0_strength)


def generate_first_img(cfg, state, img, strength):
    model = state.ddim_v_sampler.model
    height, width, _ = img.shape
    img_ = src.img_util.numpy2tensor(img)

    num_samples = 1

    encoder_posterior = model.encode_first_stage(img_.to(global_state.device))
    x0 = model.get_first_stage_encoding(encoder_posterior).detach()

    detected_map = state.detector(img)
    detected_map = ControlNet.annotator.util.HWC3(detected_map)

    control = torch.from_numpy(detected_map.copy()).float().to(global_state.device) / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    cond = {
        'c_concat': [control],
        'c_crossattn': [
            model.get_learned_conditioning(
                [cfg.prompt + ', ' + cfg.a_prompt] * num_samples)
        ]
    }
    un_cond = {
        'c_concat': [control],
        'c_crossattn':
            [model.get_learned_conditioning([cfg.n_prompt] * num_samples)]
    }
    shape = (4, height // 8, width // 8)

    state.controller.set_task('initfirst')
    pytorch_lightning.seed_everything(cfg.seed)

    samples, _ = state.ddim_v_sampler.sample(
        cfg.ddim_steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        eta=0.0,
        unconditional_guidance_scale=cfg.scale,
        unconditional_conditioning=un_cond,
        controller=state.controller,
        x0=x0,
        strength=strength
    )
    x_samples = model.decode_first_stage(samples)
    x_smaples_normalized = einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5
    x_samples_np = x_smaples_normalized.cpu().numpy().clip(0, 255).astype(numpy.uint8)
    return x_samples, x_samples_np


def process2():
    cfg = get_config()

    state = global_state.GlobalState()
    state.update_sd_model(cfg.sd_model, cfg.control_type)
    state.update_detector(cfg.control_type, cfg.canny_low, cfg.canny_high)
    state.processing_state = global_state.ProcessingState.KEY_IMGS

    # reset key dir
    shutil.rmtree(cfg.key_dir)
    os.makedirs(cfg.key_dir, exist_ok=True)

    model = state.ddim_v_sampler.model
    model.control_scales = [cfg.control_strength] * 13

    num_samples = 1
    eta = 0.0
    firstx0 = True
    pixelfusion = cfg.use_mask
    imgs = sorted(os.listdir(cfg.input_dir))
    imgs = [os.path.join(cfg.input_dir, img) for img in imgs]

    first_result = state.first_result
    first_img = state.first_img
    pre_result = first_result
    pre_img = first_img

    for i in range(0, cfg.frame_count - 1, cfg.interval):
        cid = i + 1
        frame = cv2.imread(imgs[i + 1])
        print(cid)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ControlNet.annotator.util.HWC3(frame)
        H, W, C = img.shape

        if cfg.color_preserve or state.color_corrections is None:
            img_ = src.img_util.numpy2tensor(img)
        else:
            img_ = apply_color_correction(state.color_corrections,
                                          PIL.Image.fromarray(img))
            img_ = transforms.PILToTensor()(img_).unsqueeze(0)[:, :3] / 127.5 - 1
        encoder_posterior = model.encode_first_stage(img_.to(global_state.device))
        x0 = model.get_first_stage_encoding(encoder_posterior).detach()

        detected_map = state.detector.detector(img)
        detected_map = ControlNet.annotator.util.HWC3(detected_map)

        control = torch.from_numpy(
            detected_map.copy()).float().to(global_state.device) / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        cond = {
            'c_concat': [control],
            'c_crossattn': [
                model.get_learned_conditioning(
                    [cfg.prompt + ', ' + cfg.a_prompt] * num_samples)
            ]
        }
        un_cond = {
            'c_concat': [control],
            'c_crossattn':
                [model.get_learned_conditioning([cfg.n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        cond['c_concat'] = [control]
        un_cond['c_concat'] = [control]

        image1 = torch.from_numpy(pre_img).permute(2, 0, 1).float()
        image2 = torch.from_numpy(img).permute(2, 0, 1).float()
        warped_pre, bwd_occ_pre, bwd_flow_pre = flow.flow_utils.get_warped_and_mask(
            state.flow_model, image1, image2, pre_result, False)
        blend_mask_pre = blur(
            functional.max_pool2d(bwd_occ_pre, kernel_size=9, stride=1, padding=4))
        blend_mask_pre = torch.clamp(blend_mask_pre + bwd_occ_pre, 0, 1)

        image1 = torch.from_numpy(first_img).permute(2, 0, 1).float()
        warped_0, bwd_occ_0, bwd_flow_0 = flow.flow_utils.get_warped_and_mask(
            state.flow_model, image1, image2, first_result, False)
        blend_mask_0 = blur(
            functional.max_pool2d(bwd_occ_0, kernel_size=9, stride=1, padding=4))
        blend_mask_0 = torch.clamp(blend_mask_0 + bwd_occ_0, 0, 1)

        if firstx0:
            mask = 1 - functional.max_pool2d(blend_mask_0, kernel_size=8)
            state.controller.set_warp(
                functional.interpolate(bwd_flow_0 / 8.0,
                                       scale_factor=1. / 8,
                                       mode='bilinear'), mask)
        else:
            mask = 1 - functional.max_pool2d(blend_mask_pre, kernel_size=8)
            state.controller.set_warp(
                functional.interpolate(bwd_flow_pre / 8.0,
                                       scale_factor=1. / 8,
                                       mode='bilinear'), mask)

        state.controller.set_task('keepx0, keepstyle')
        pytorch_lightning.seed_everything(cfg.seed)
        samples, intermediates = state.ddim_v_sampler.sample(
            cfg.ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=cfg.scale,
            unconditional_conditioning=un_cond,
            controller=state.controller,
            x0=x0,
            strength=1 - cfg.x0_strength)
        direct_result = model.decode_first_stage(samples)

        if not pixelfusion:
            pre_result = direct_result
            pre_img = img
            viz = (
                    einops.rearrange(direct_result, 'b c h w -> b h w c') * 127.5 +
                    127.5).cpu().numpy().clip(0, 255).astype(numpy.uint8)

        else:

            blend_results = (1 - blend_mask_pre
                             ) * warped_pre + blend_mask_pre * direct_result
            blend_results = (
                                    1 - blend_mask_0) * warped_0 + blend_mask_0 * blend_results

            bwd_occ = 1 - torch.clamp(1 - bwd_occ_pre + 1 - bwd_occ_0, 0, 1)
            blend_mask = blur(
                functional.max_pool2d(bwd_occ, kernel_size=9, stride=1, padding=4))
            blend_mask = 1 - torch.clamp(blend_mask + bwd_occ, 0, 1)

            encoder_posterior = model.encode_first_stage(blend_results)
            xtrg = model.get_first_stage_encoding(
                encoder_posterior).detach()  # * mask
            blend_results_rec = model.decode_first_stage(xtrg)
            encoder_posterior = model.encode_first_stage(blend_results_rec)
            xtrg_rec = model.get_first_stage_encoding(
                encoder_posterior).detach()
            xtrg_ = (xtrg + 1 * (xtrg - xtrg_rec))  # * mask
            blend_results_rec_new = model.decode_first_stage(xtrg_)
            tmp = (abs(blend_results_rec_new - blend_results).mean(
                dim=1, keepdims=True) > 0.25).float()
            mask_x = functional.max_pool2d((functional.interpolate(tmp,
                                                                   scale_factor=1 / 8.,
                                                                   mode='bilinear') > 0).float(),
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)

            mask = (1 - functional.max_pool2d(1 - blend_mask, kernel_size=8)
                    )  # * (1-mask_x)

            if cfg.smooth_boundary:
                noise_rescale = src.img_util.find_flat_region(mask)
            else:
                noise_rescale = torch.ones_like(mask)
            masks = []
            for i in range(cfg.ddim_steps):
                if i <= cfg.ddim_steps * cfg.mask_period[
                    0] or i >= cfg.ddim_steps * cfg.mask_period[1]:
                    masks += [None]
                else:
                    masks += [mask * cfg.mask_strength]

            # mask 3
            # xtrg = ((1-mask_x) *
            #         (xtrg + xtrg - xtrg_rec) + mask_x * samples) * mask
            # mask 2
            # xtrg = (xtrg + 1 * (xtrg - xtrg_rec)) * mask
            xtrg = (xtrg + (1 - mask_x) * (xtrg - xtrg_rec)) * mask  # mask 1

            tasks = 'keepstyle, keepx0'
            if not firstx0:
                tasks += ', updatex0'
            if i % cfg.style_update_freq == 0:
                tasks += ', updatestyle'
            state.controller.set_task(tasks, 1.0)

            pytorch_lightning.seed_everything(cfg.seed)
            samples, _ = state.ddim_v_sampler.sample(
                cfg.ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=cfg.scale,
                unconditional_conditioning=un_cond,
                controller=state.controller,
                x0=x0,
                strength=1 - cfg.x0_strength,
                xtrg=xtrg,
                mask=masks,
                noise_rescale=noise_rescale)
            x_samples = model.decode_first_stage(samples)
            pre_result = x_samples
            pre_img = img

            viz = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
                   127.5).cpu().numpy().clip(0, 255).astype(numpy.uint8)

        PIL.Image.fromarray(viz[0]).save(
            os.path.join(cfg.key_dir, f'{cid:04d}.png'))

    key_video_path = os.path.join(cfg.work_dir, 'key.mp4')
    fps = src.video_util.get_fps(cfg.input_path)
    fps //= cfg.interval
    src.video_util.frame_to_video(key_video_path, cfg.key_dir, fps, False)

    return key_video_path


def get_config() -> src.config.RerenderConfig:
    return src.config.RerenderConfig().create_from_parameters(
        "input.mp4",
        "output.mp4",
        "watercolor painting",
        work_dir=None,
        key_subdir='keys',
        frame_count=None,
        interval=10,
        crop=(0, 0, 0, 0),
        sd_model=None,
        a_prompt='',
        n_prompt='',
        ddim_steps=20,
        scale=7.5,
        control_type='HED',
        control_strength=1,
        seed=123,
        image_resolution=512,
        x0_strength=-1,
        style_update_freq=10,
        cross_period=(0, 1),
        warp_period=(0, 0.1),
        mask_period=(0.5, 0.8),
        ada_period=(1.0, 1.0),
        mask_strength=0.5,
        inner_strength=0.9,
        smooth_boundary=True,
        color_preserve=True,
    )


def setup_color_correction(image):
    correction_target = cv2.cvtColor(numpy.asarray(image.copy()), cv2.COLOR_RGB2LAB)
    return correction_target


def apply_color_correction(correction, original_image):
    image = PIL.Image.fromarray(
        cv2.cvtColor(
            skimage.exposure.match_histograms(
                cv2.cvtColor(numpy.asarray(original_image), cv2.COLOR_RGB2LAB),
                correction,
                channel_axis=2),
            cv2.COLOR_LAB2RGB).astype('uint8'))

    image = blendmodes.blend.blendLayers(image, original_image, blendmodes.blend.BlendType.LUMINOSITY)

    return image

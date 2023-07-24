# noinspection PyPackageRequirements
import accelerate.utils
# noinspection PyUnresolvedReferences
import blendmodes.blend
# noinspection PyPackageRequirements
import einops
import numpy
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
import moviepy.video.io.ffmpeg_writer


def main(cfg: src.config.RerenderConfig):
    state = get_state(cfg)

    # noinspection PyUnresolvedReferences
    import decord
    reader = decord.VideoReader(cfg.input_path, width=cfg.image_resolution, height=cfg.image_resolution)

    first_image = reader.next().asnumpy()
    first_result = generate_first_result(state, cfg, first_image)

    previous_image = first_image
    previous_result = first_result

    writer = moviepy.video.io.ffmpeg_writer.FFMPEG_VideoWriter(
        cfg.output_path,
        (cfg.image_resolution, cfg.image_resolution),
        reader.get_avg_fps() / cfg.interval,
        ffmpeg_params=["-crf", "15", "-metadata", "title=Rerender A Video"],
    )

    for i in range(0, 16, cfg.interval):
        reader.seek(0)
        image = reader.next().asnumpy()

        result = generate_next_image(
            state, cfg, first_image, first_result, previous_image, previous_result, i, image)

        writer.write_frame(torch_to_numpy(image)[0])

        previous_image = image
        previous_result = result

    writer.close()


def get_state(cfg: src.config.RerenderConfig):
    state = global_state.GlobalState()
    state.update_sd_model(cfg.sd_model, cfg.control_type)
    state.update_controller(cfg.inner_strength, cfg.mask_period, cfg.cross_period, cfg.ada_period, cfg.warp_period)
    state.update_detector(cfg.control_type, cfg.canny_low, cfg.canny_high)
    state.processing_state = global_state.ProcessingState.FIRST_IMG

    control_net = state.ddim_v_sampler.model
    control_net.control_scales = [cfg.control_strength] * 13
    control_net.cond_stage_model.device = global_state.device
    control_net.to(global_state.device)
    return state


def generate_first_result(state: global_state.GlobalState, cfg: src.config.RerenderConfig,
                          input_image: numpy.ndarray) -> torch.Tensor:
    control_net = state.ddim_v_sampler.model
    height, width, _ = input_image.shape
    tensor_image = src.img_util.numpy2tensor(input_image)

    num_samples = 1

    encoder_posterior = control_net.encode_first_stage(tensor_image.to(global_state.device))
    x0 = control_net.get_first_stage_encoding(encoder_posterior).detach()

    detected_map = state.detector(input_image)
    detected_map = ControlNet.annotator.util.HWC3(detected_map)

    control = torch.from_numpy(detected_map.copy()).float().to(global_state.device) / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    conditioning = {
        'c_concat': [control],
        'c_crossattn': [
            control_net.get_learned_conditioning([cfg.prompt + ', ' + cfg.a_prompt] * num_samples)
        ]
    }
    unconditional_conditioning = {
        'c_concat': [control],
        'c_crossattn': [control_net.get_learned_conditioning([cfg.n_prompt] * num_samples)]
    }
    shape = (4, height // 8, width // 8)

    state.controller.set_task('initfirst')
    accelerate.utils.set_seed(cfg.seed)

    samples, _ = state.ddim_v_sampler.sample(
        cfg.ddim_steps,
        num_samples,
        shape,
        conditioning=conditioning,
        verbose=False,
        unconditional_guidance_scale=cfg.scale,
        unconditional_conditioning=unconditional_conditioning,
        controller=state.controller,
        x0=x0,
        strength=1 - cfg.x0_strength
    )
    return control_net.decode_first_stage(samples)


def generate_next_image(
        state: global_state.GlobalState,
        cfg: src.config.RerenderConfig,
        first_image: numpy.ndarray,
        first_result: torch.Tensor,
        previous_image: numpy.ndarray,
        previous_result: torch.Tensor,
        i: int,
        image: numpy.ndarray,
):
    control_net = state.ddim_v_sampler.model

    num_samples = 1

    blur = transforms.GaussianBlur(kernel_size=(9, 9), sigma=(18, 18))

    height, width, _ = image.shape

    tensor_image = src.img_util.numpy2tensor(image)

    encoder_posterior = control_net.encode_first_stage(tensor_image.to(global_state.device))
    x0 = control_net.get_first_stage_encoding(encoder_posterior).detach()

    detected_map = state.detector(image)
    detected_map = ControlNet.annotator.util.HWC3(detected_map)

    control = torch.from_numpy(detected_map.copy()).float().to(global_state.device) / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    cond = {
        'c_concat': [control],
        'c_crossattn': [
            control_net.get_learned_conditioning([cfg.prompt + ', ' + cfg.a_prompt] * num_samples)
        ]
    }
    un_cond = {
        'c_concat': [control],
        'c_crossattn': [control_net.get_learned_conditioning([cfg.n_prompt] * num_samples)]
    }
    shape = (4, height // 8, width // 8)

    cond['c_concat'] = [control]
    un_cond['c_concat'] = [control]

    image1 = torch.from_numpy(previous_image).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image).permute(2, 0, 1).float()
    warped_pre, bwd_occ_pre, bwd_flow_pre = flow.flow_utils.get_warped_and_mask(
        state.flow_model, image1, image2, previous_result, False
    )
    blend_mask_pre = blur(functional.max_pool2d(bwd_occ_pre, kernel_size=9, stride=1, padding=4))
    blend_mask_pre = torch.clamp(blend_mask_pre + bwd_occ_pre, 0, 1)

    image1 = torch.from_numpy(first_image).permute(2, 0, 1).float()
    warped_0, bwd_occ_0, bwd_flow_0 = flow.flow_utils.get_warped_and_mask(
        state.flow_model, image1, image2, first_result, False
    )
    blend_mask_0 = blur(functional.max_pool2d(bwd_occ_0, kernel_size=9, stride=1, padding=4))
    blend_mask_0 = torch.clamp(blend_mask_0 + bwd_occ_0, 0, 1)

    mask = 1 - functional.max_pool2d(blend_mask_0, kernel_size=8)
    state.controller.set_warp(
        functional.interpolate(bwd_flow_0 / 8.0, scale_factor=1. / 8, mode='bilinear'),
        mask
    )

    state.controller.set_task('keepx0, keepstyle')
    accelerate.utils.set_seed(cfg.seed)
    samples, intermediates = state.ddim_v_sampler.sample(
        cfg.ddim_steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        unconditional_guidance_scale=cfg.scale,
        unconditional_conditioning=un_cond,
        controller=state.controller,
        x0=x0,
        strength=1 - cfg.x0_strength
    )
    direct_result = control_net.decode_first_stage(samples)

    if not cfg.use_mask:
        return direct_result

    else:

        blend_results = (1 - blend_mask_pre) * warped_pre + blend_mask_pre * direct_result
        blend_results = (1 - blend_mask_0) * warped_0 + blend_mask_0 * blend_results

        bwd_occ = 1 - torch.clamp(1 - bwd_occ_pre + 1 - bwd_occ_0, 0, 1)
        blend_mask = blur(functional.max_pool2d(bwd_occ, kernel_size=9, stride=1, padding=4))
        blend_mask = 1 - torch.clamp(blend_mask + bwd_occ, 0, 1)

        encoder_posterior = control_net.encode_first_stage(blend_results)
        xtrg = control_net.get_first_stage_encoding(encoder_posterior).detach()  # * mask
        blend_results_rec = control_net.decode_first_stage(xtrg)
        encoder_posterior = control_net.encode_first_stage(blend_results_rec)
        xtrg_rec = control_net.get_first_stage_encoding(encoder_posterior).detach()
        xtrg_ = (xtrg + 1 * (xtrg - xtrg_rec))  # * mask
        blend_results_rec_new = control_net.decode_first_stage(xtrg_)
        tmp = (abs(blend_results_rec_new - blend_results).mean(
            dim=1, keepdims=True) > 0.25).float()
        mask_x = functional.max_pool2d(
            (functional.interpolate(tmp, scale_factor=1 / 8., mode='bilinear') > 0).float(),
            kernel_size=3,
            stride=1,
            padding=1)

        mask = (1 - functional.max_pool2d(1 - blend_mask, kernel_size=8))  # * (1-mask_x)

        if cfg.smooth_boundary:
            noise_rescale = src.img_util.find_flat_region(mask)
        else:
            noise_rescale = torch.ones_like(mask)
        masks = []
        for i2 in range(cfg.ddim_steps):
            if i2 <= cfg.ddim_steps * cfg.mask_period[0] or i2 >= cfg.ddim_steps * cfg.mask_period[1]:
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

        if i % cfg.style_update_freq == 0:
            tasks += ', updatestyle'

        state.controller.set_task(tasks, 1.0)

        accelerate.utils.set_seed(cfg.seed)
        samples, _ = state.ddim_v_sampler.sample(
            cfg.ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            unconditional_guidance_scale=cfg.scale,
            unconditional_conditioning=un_cond,
            controller=state.controller,
            x0=x0,
            strength=1 - cfg.x0_strength,
            xtrg=xtrg,
            mask=masks,
            noise_rescale=noise_rescale
        )

        return control_net.decode_first_stage(samples)


def torch_to_numpy(a: torch.Tensor) -> numpy.ndarray:
    samples_normalized = einops.rearrange(a, 'b c h w -> b h w c') * 127.5 + 127.5
    return samples_normalized.cpu().numpy().clip(0, 255).astype(numpy.uint8)


def get_config(input_, output, prompt) -> src.config.RerenderConfig:
    a = src.config.RerenderConfig()
    a.create_from_parameters(
        input_,
        output,
        prompt,
        work_dir=None,
        key_subdir='keys',
        frame_count=None,
        interval=10,
        crop=(0, 0, 0, 0),
        sd_model='Stable Diffusion 1.5',
        a_prompt='',
        n_prompt='',
        ddim_steps=20,
        scale=7.5,
        control_type='HED',
        control_strength=1,
        seed=123,
        image_resolution=512,
        x0_strength=1,
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
    return a

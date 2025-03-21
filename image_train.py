"""
Train a diffusion model on images.
"""

import os
import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    process_argements,
)
from guided_diffusion.train_util import TrainLoop


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = create_argparser().parse_args()
    process_argements(args)
    dist_util.setup_dist()
    logger.configure(dir=args.save_dir, format_strs=["stdout", "log", "tensorboard", 'csv'])

    logger.log("creating model and diffusion...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    print(args)
    logger.log("creating data loader...")
    data = load_data(
        dataset_mode=args.dataset_mode,
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        coarse_path=args.coarse_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        coarse_cond=args.coarse_cond,
        is_train=args.is_train,
        shifted=args.shifted,
        args=args
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        large_size=args.large_size,
        small_size=args.small_size,
        num_classes=args.num_classes,
        compression_type=args.compression_type,
        compression_level=args.compression_level,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        drop_rate=args.drop_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        save_dir=args.save_dir,
        args=args
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        schedule_sampler="uniform",
        large_size=64,
        small_size=32,
        compression_type="resize",
        compression_level=0,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        drop_rate=0.0,
        log_interval=50,
        save_interval=500,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        is_train=True,
        save_dir="",
        shifted=False,  # 是否基于偏移的 SSM 进行生成，即基于视频上一帧的低分辨图像和这一帧的 SSM 生成这一帧的图像
        condition="ssm",  # 条件类型，可取值为 ssm | sketch | layout
        ssm_path="",  # 条件的路径，为方便起见将所有条件都称为 ssm
        random_eliminate=False,  # 随机对语义区域进行掩蔽
        eliminate_level=0,  # 掩蔽语义区域的等级，仅在 random_eliminate=False 时起效
        random_sample=False,  # 随机对输入图像进行颜色采样
        sample_level=-1,  # 颜色采样等级，每个等级表示 0.01%，-1 表示不进行采样
        eliminate_channels_assist=False,  # 在掩蔽时，是否用掩蔽信息辅助生成
        other_folder="",  # 其他路径，在多层级语义中指完整语义分割图的路径，以模拟 layout 结合 boundary 的效果
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

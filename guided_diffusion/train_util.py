import copy
import functools
import os
import time
import random
import cv2
import numpy as np
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import torchvision as tv
print(f"cuda available = {th.cuda.is_available()}")
from torch.nn.functional import interpolate
import torch

# import dist_util, logger
# from fp16_util import MixedPrecisionTrainer
# from nn import update_ema
# from resample import LossAwareSampler, UniformSampler

### Prima erano così
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        large_size,
        small_size,
        compression_type,
        compression_level,
        num_classes,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        drop_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        save_dir="",
        args=None
    ):
        self.args = args
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.large_size = large_size
        self.small_size = small_size
        self.compression_type = compression_type
        self.compression_level = compression_level
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.drop_rate = drop_rate
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.save_dir = save_dir if save_dir else get_blob_logdir()
        self.image_save_folder = os.path.join(self.save_dir, "images")
        self.compressed_save_folder = os.path.join(self.save_dir, "compressed")
        self.labels_save_folder = os.path.join(self.save_dir, "labels")
        self.samples_save_folder = os.path.join(self.save_dir, "samples")
        os.makedirs(self.image_save_folder, exist_ok=True)
        os.makedirs(self.compressed_save_folder, exist_ok=True)
        os.makedirs(self.labels_save_folder, exist_ok=True)
        os.makedirs(self.samples_save_folder, exist_ok=True)
        self.prefix = self.save_dir.strip("/").split("/")[-1]

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                try:
                    self.model.load_state_dict(
                        th.load(
                            resume_checkpoint, map_location=dist_util.dev()
                        )
                    )
                except RuntimeError as e:
                    print(e)
                    self.model.load_state_dict(
                        th.load(
                            resume_checkpoint, map_location=dist_util.dev()
                        ), strict=False
                    )
                    print("Loading done (not strict)")

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = th.load(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

            if self.opt.param_groups[0]['lr'] != self.lr:
                self.opt.param_groups[0]['lr'] = self.lr

    def run_loop(self):
        temp_folder = "/home/Users/dqy/Projects/SPIC/temp/"
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond_ori = next(self.data)
            if batch is None:
                self.step += 1
                continue
            cond = self.preprocess_input(cond_ori)
            if 'coarse' in cond:
                cond["compressed"] = cond["coarse"]
            else:
                if self.compression_type == 'down+bpg':
                    image = interpolate(batch, (self.small_size, self.large_size), mode="area") 
                    bpg_image_list = []
                    for i in range(image.shape[0]):
                        timestamp = time.time()
                        cv2.imwrite(f"{temp_folder}compressed_bpg-{self.prefix}#{timestamp}.png", cv2.cvtColor((image[i].cpu().numpy().transpose(1, 2, 0)+1)/2.*255.0, cv2.COLOR_BGR2RGB))
                        os.system(f"/home/Users/dqy/myLibs/libbpg-0.9.7/bin/bpgenc -c ycbcr -q  {int(self.compression_level)} -o {temp_folder}compressed_bpg-{self.prefix}#{timestamp}.bpg {temp_folder}compressed_bpg-{self.prefix}#{timestamp}.png")
                        os.system(f"/home/Users/dqy/myLibs/libbpg-0.9.7/bin/bpgdec -o {temp_folder}compressed_bpg-{self.prefix}#{timestamp}.png {temp_folder}compressed_bpg-{self.prefix}#{timestamp}.bpg")
                        decompressed_image = cv2.imread(f"{temp_folder}compressed_bpg-{self.prefix}#{timestamp}.png")
                        os.remove(f"{temp_folder}compressed_bpg-{self.prefix}#{timestamp}.png")
                        os.remove(f"{temp_folder}compressed_bpg-{self.prefix}#{timestamp}.bpg")
                        decompressed_image = cv2.cvtColor(decompressed_image, cv2.COLOR_BGR2RGB)
                        tensor_image = (th.from_numpy(decompressed_image).permute(2, 0, 1)*2/255.0)-1
                        bpg_image_list.append(tensor_image)
                    compressed = th.stack(bpg_image_list)
                    cond['compressed'] = compressed.cuda()
                elif self.compression_type == 'down':
                    cond['compressed'] = interpolate(batch, (self.small_size, self.large_size), mode="area") 
                elif self.compression_type == 'sample':  # 像素点采样
                    if self.args.random_sample:
                        sample_ratio = random.randint(0, 30) * 0.0001
                    elif self.args.sample_level != -1:
                        sample_ratio = self.args.sample_level * 0.0001
                    else:
                        raise AttributeError("Must assign random_sample of sample_level!")
                    H, W = batch.shape[2:]
                    sample_count = int(H * W * sample_ratio)
                    sampled = random_sample_pixels(batch, sample_count)
                    cond['compressed'] = sampled
                else:
                    cond['compressed'] = interpolate(batch, (self.small_size, self.large_size), mode="area") 
            self.run_step(batch, cond)
            
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0 and self.step != 0:
            # if self.step != 0:
                self.save()
                # sample_fn = (
                #     self.diffusion.p_sample_loop
                # )
                # logger.log(f"Sampling...")
                # self.model.eval()
                # with torch.no_grad():
                #     sample = sample_fn(
                #         self.model,
                #         (self.args.batch_size, 3, batch.shape[2], batch.shape[3]),
                #         clip_denoised=True,
                #         model_kwargs=cond,
                #         progress=True
                #     )
                # # sample = batch
                # compressed_img = cond['compressed']
                # logger.log(f"Saving images...")
                # label = (cond_ori['label_ori'].float() / 255.0)
                # path = cond_ori['path']
                # self.save_images(batch, compressed_img, sample, label, self.step, path)
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond): 
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            

            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                
                with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                    th.save(state_dict, f)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(self.save_dir, f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

        # I want to delete all the .pt files except the three just saved
        # create the list of files to keep
        files_to_keep = ["log.txt", "progress.csv", f"model{(self.step+self.resume_step):06d}.pt", f"opt{(self.step+self.resume_step):06d}.pt"]

        for rate, params in zip(self.ema_rate, self.ema_params):
            files_to_keep.append(f"ema_{rate}_{(self.step+self.resume_step):06d}.pt")
        files_to_keep.append('tb')
            
        # get the list of files in the directory
        files_in_dir = os.listdir(self.save_dir)
        # delete all the files that are not in the list of files to keep
        logger.log("Deleting old checkpoints...")
        # for file in files_in_dir:
        #     if file not in files_to_keep:
        #         os.remove(os.path.join(self.save_dir, file))

    def save_images(self, image, compressed_img, sample, label, step, path):
        for j in range(image.shape[0]):
            save_compressed(((compressed_img[j] + 1.0) / 2.0),
                            os.path.join(self.compressed_save_folder, f"Step{step:06d}-" + path[j].split('/')[-1].split('.')[0]), self.args)
            tv.utils.save_image(((image[j] + 1.0) / 2.0),
                                os.path.join(self.image_save_folder, f"Step{step:06d}-" + path[j].split('/')[-1].split('.')[0] + '.png'))
            tv.utils.save_image(sample[j],
                                os.path.join(self.samples_save_folder, f"Step{step:06d}-" + path[j].split('/')[-1].split('.')[0] + '.png'))
            if self.args.condition != "layout":
                tv.utils.save_image(label[j] * 255.0 / 35.0,  # 0~18 的标签和 255，保存为以 7 为灰度间隔方便可视化
                                    os.path.join(self.labels_save_folder, f"Step{step:06d}-" + path[j].split('/')[-1].split('.')[0] + '.png'))

    def preprocess_input(self, cond_):#, compressed):
        # move to GPU and change data types
        if self.args.condition in ["ssm", "boundary", "sketch"]:
            cond_['label'] = cond_['label'].long()

        # create one-hot label map
        label_map = cond_['label']
        eliminate_channels_cond = cond_['eliminate_semantics'][:, :, None, None].long()
        bs, _, h, w = label_map.size()
        if self.num_classes == 19:  # 对于 Cityscapes 数据集的语义分割图，由于先天存在忽略区域，因此通道数始终增加 1
            nc = self.num_classes+1
        elif self.args.random_eliminate:  # 对于 Cityscapes 数据集的其他语义表征和其他数据集，使用随机忽略语义区域时，通道数增加 1
            nc = self.num_classes + 1
        else:
            nc = self.num_classes
        if self.args.condition in ["ssm", "boundary", "sketch"]:
            input_label = th.FloatTensor(bs, nc, h, w).zero_()
            eliminate_channels = eliminate_channels_cond.view(bs, nc, 1, 1)
            input_semantics = input_label.scatter_(1, label_map, 1.0)
            if self.num_classes == 19:
                input_semantics = input_semantics[:, :-1, :, :]
                eliminate_channels = eliminate_channels[:, :-1, :, :]

            # concatenate instance map if it exists
            if 'instance' in cond_:
                inst_map = cond_['instance']
                instance_edge_map = self.get_edges(inst_map)
                input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)

            if self.drop_rate > 0.0:
                mask = (th.rand([input_semantics.shape[0], 1, 1, 1]) > self.drop_rate).float()
                input_semantics = input_semantics * mask
        else:
            assert self.args.condition == "layout"
            input_semantics = label_map
            eliminate_channels = eliminate_channels_cond
        cond = {key: value for key, value in cond_.items() if key not in ['label', 'instance', 'path', 'label_ori']}
        cond['y'] = input_semantics
        cond['eliminate_channels'] = eliminate_channels
        
        
        return cond

    def get_edges(self, t):
        edge = th.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def random_sample_pixels(images, sample_count):
    """
    随机选择 sample_count 个像素点保留原值，其余像素点置为 0

    :param images: 输入的图像批次，形状为 (B, 3, H, W)
    :param sample_count: 要保留的随机像素点数量
    :return: 处理后的图像批次
    """
    B, _, H, W = images.shape
    total_pixels = H * W
    
    # 对每张图像进行处理
    processed_images = th.zeros_like(images)

    for b in range(B):
        # 随机选择 sample_count 个像素点的索引
        random_indices = np.random.choice(total_pixels, sample_count, replace=False)
        
        # 将保留的像素点位置恢复为原值
        h_indices = random_indices // W  # 计算行索引
        w_indices = random_indices % W   # 计算列索引
        
        # 将原图像的保留像素点赋值给新图像
        processed_images[b, :, h_indices, w_indices] = images[b, :, h_indices, w_indices]

    return processed_images


def save_compressed(compressed, path, args):
    if args.compression_type == 'down+bpg':
        tv.utils.save_image(compressed, path + '.png')
        os.system(
            f"/home/Users/dqy/myLibs/libbpg-0.9.7/bin/bpgenc -c ycbcr -q  {int(args.compression_level)} -o {path}.bpg {path}.png")
        os.system(f"/home/Users/dqy/myLibs/libbpg-0.9.7/bin/bpgdec -o {path}.png {path}.bpg")
    else:
        tv.utils.save_image(compressed, path + '.png')

import h5py
from PIL import Image
import scipy.io
import argparse, os
import pandas as pd
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import trange
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
import sys

sys.path.append("../utils/")
from nsd_access.nsda import NSDAccess
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler  


class LightConfidenceNet(torch.nn.Module):
    def __init__(self, feat_dim=77 * 768):
        super().__init__()
     
        self.estimator = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 4, bias=False),  
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4, 1, bias=False),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.estimator(x.flatten(start_dim=1))


def load_model_from_config(config, ckpt, gpu, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda(f"cuda:{gpu}")
    model.eval()
    return model


def load_img_from_arr(img_arr):
    image = Image.fromarray(img_arr).convert("RGB")
    w, h = 512, 512
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--imgidx",
        required=True,
        type=int,
        help="img idx"
    )
    parser.add_argument(
        "--gpu",
        required=True,
        type=int,
        help="gpu"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--subject",
        required=True,
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )
    parser.add_argument(
        "--method",
        required=True,
        type=str,
        help="cvpr or text or gan",
    )
    opt = parser.parse_args()
    seed_everything(opt.seed)
    imgidx = opt.imgidx
    gpu = opt.gpu
    method = opt.method
    subject = opt.subject

    gandir = f'../../'
    captdir = f'../../'
    nsd_expdesign = scipy.io.loadmat('../../nsd/nsddata/experiments/nsd/nsd_expdesign.mat')
    sharedix = nsd_expdesign['sharedix'] - 1

    nsda = NSDAccess('../../nsd/')
    sf = h5py.File(nsda.stimuli_file, 'r')
    sdataset = sf.get('imgBrick')

    stims_ave = np.load(f'../../mrifeat73000/{subject}/{subject}_stims_ave.npy')

    tr_idx = np.zeros_like(stims_ave)
    for idx, s in enumerate(stims_ave):
        if s in sharedix:
            tr_idx[idx] = 0
        else:
            tr_idx[idx] = 1

    # Load Stable Diffusion Model
    config = './.yaml'
    ckpt = './.ckpt'
    config = OmegaConf.load(f"{config}")
    torch.cuda.set_device(gpu)

    model = load_model_from_config(config, f"{ckpt}", gpu)

    n_samples = 1
    ddim_steps = 50
    ddim_eta = 0.0
    strength = 0.8
    scale = 5.0
    n_iter = 4
    precision = 'autocast'
    precision_scope = autocast if precision == "autocast" else nullcontext
    batch_size = n_samples
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    confidence_net = None  

    outdir = f'../../'
    os.makedirs(outdir, exist_ok=True)

    sample_path = os.path.join(outdir, f"samples")
    os.makedirs(sample_path, exist_ok=True)
    precision = 'autocast'
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    # Load z (Image)
    imgidx_te = np.where(tr_idx == 0)[0][imgidx]  # Extract test image index
    idx73k = stims_ave[imgidx_te]
    Image.fromarray(np.squeeze(sdataset[idx73k, :, :, :]).astype(np.uint8)).save(
        os.path.join(sample_path, f"{imgidx:05}_org.png"))

    print(f"Model is on device: {next(model.parameters()).device}")

    print(f"Method selected: {method}")


    roi_latent = 'early'
    scores_latent = np.load(f'../../')
    imgarr = torch.Tensor(scores_latent[imgidx, :].reshape(4, 40, 40)).unsqueeze(0).to('cuda')

        # Generate image from Z
    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                print(f"imgarr shape: {imgarr.shape}, dtype: {imgarr.dtype}")
                try:
                    x_samples = model.decode_first_stage(imgarr)
                except Exception as e:
                    print(f"Error during decoding: {e}")
                    return
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
    im = Image.fromarray(x_sample.astype(np.uint8)).resize((512, 512))
    im = np.array(im)


    init_image = load_img_from_arr(im).to('cuda')
 
    roi_c = 'ventral'
    scores_c = np.load(f'../../decoded73000_zhixing/{subject}/{subject}_{roi_c}_scores_c.npy')
    carr = scores_c[imgidx, :].reshape(77, 768)


 
import torch
import os
import numpy as np
from PIL import Image
from einops import rearrange
from tqdm import trange
from contextlib import contextmanager

def dcaf(carr, confidence_net, model, sampler, init_latent, t_enc, scale, sample_path,
         imgidx=0, n_iter=10, batch_size=1, weight_multiplier=1.0, device='cuda'):

    # 将 carr 转换为张量，并传入 CUDA
    c = torch.Tensor(carr).unsqueeze(0).to(device)

    # 获取置信度权重
    with torch.no_grad():
        conf_weight = confidence_net(c)  # 输出形状应该为 (1, feature_dim)
    conf_weight = conf_weight * weight_multiplier

    base_count = 0

    with torch.inference_mode():
        with precision_scope(device):
            with model.ema_scope():
                for n in trange(n_iter, desc="Sampling"):

                    time_decay = max(0.0, 1.0 - (n / n_iter))
                    
                    # 计算动态条件（Dynamic Conditioning）
                    uc = model.get_learned_conditioning([""])
                    dynamic_c = c * conf_weight * time_decay + uc * (1 - conf_weight * time_decay)

                    uc = model.get_learned_conditioning(batch_size * [""])
                    z_enc = sampler.stochastic_encode(
                        init_latent,
                        torch.tensor([t_enc] * batch_size).to(device)
                    )
                    samples = sampler.decode(
                        z_enc, dynamic_c, t_enc,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc
                    )
                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples:
                        print(f"x_samples类型: {type(x_samples)}, 形状: {x_samples.shape}")
                        x_sample_np = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample_np.astype(np.uint8))
                        img.save(os.path.join(sample_path, f"{imgidx:05}_{base_count:03}.png"))
                        base_count += 1

@contextmanager
def precision_scope(device):
    yield

if __name__ == "__main__":
    dcaf()

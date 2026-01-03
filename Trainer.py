import os
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.distributed as dist

from model.loss import LapLoss
from config import *


class Model:
    def __init__(self, local_rank):
        backbonetype, multiscaletype = MODEL_CONFIG['MODEL_TYPE']
        backbonecfg, multiscalecfg = MODEL_CONFIG['MODEL_ARCH']

        self.net = multiscaletype(backbonetype(**backbonecfg), **multiscalecfg)
        self.name = MODEL_CONFIG['LOGNAME']

        self.net.to("cuda")

        self.optimG = AdamW(self.net.parameters(), lr=2e-4, weight_decay=1e-4)
        self.lap = LapLoss()

        if dist.is_available() and dist.is_initialized():
            self.net = DDP(self.net, device_ids=[local_rank], output_device=local_rank)
        else:
            print("Using single GPU training")

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def load_model(self, name):
        ckpt_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "ckpt", f"{name}.pkl")
        )
        print("Loading checkpoint from:", ckpt_path)

        state_dict = torch.load(ckpt_path, map_location="cpu")

        filtered_state = {
            k: v for k, v in state_dict.items()
            if ("attn_mask" not in k and "HW" not in k)
        }

        missing, unexpected = self.net.load_state_dict(filtered_state, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    def save_model(self, rank=0):
        if rank == 0:
            os.makedirs("ckpt", exist_ok=True)
            torch.save(self.net.state_dict(), f"ckpt/{self.name}.pkl")

    def update(self, imgs, gt, learning_rate=0, training=True):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate

        if training:
            self.train()
            flow, mask, merged, pred = self.net(imgs)

            loss = self.lap(pred, gt).mean()
            for m in merged:
                loss += 0.5 * self.lap(m, gt).mean()

            self.optimG.zero_grad()
            loss.backward()
            self.optimG.step()

            return pred, loss
        else:
            self.eval()
            with torch.no_grad():
                flow, mask, merged, pred = self.net(imgs)
                return pred, 0

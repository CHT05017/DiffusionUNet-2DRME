import torch
from torchmetrics.aggregation import MeanMetric
from torchmetrics.functional.image import structural_similarity_index_measure

class Evaluator():
    def __init__(self, device="cuda", sync_on_compute=False):
        self.metrics = {
            "MAE": MeanMetric(sync_on_compute=sync_on_compute).to(device), # Mean Absolute Error
            "MSE": MeanMetric(sync_on_compute=sync_on_compute).to(device),
            "RMSE": MeanMetric(sync_on_compute=sync_on_compute).to(device),
            "PSNR": MeanMetric(sync_on_compute=sync_on_compute).to(device),
            "SSIM": MeanMetric(sync_on_compute=sync_on_compute).to(device),
        }

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        ### Args:
        - `preds`: .shape = (B, 1, H, W)
        - `targets`: .shape = (B, 1, H, W)
        - NOTE that all values \in [0, 1]
        """

        mini_preds = preds.min().item()
        mini_tgts = targets.min().item()

        assert mini_preds >= 0. and mini_tgts >= 0., \
            f"Check the scope of input preds and targets."
        
        err = preds - targets # (B, 1, H, W)
        eps = 1e-10
        err: torch.Tensor

        mae = err.abs().mean(dim=(1, 2, 3)) # (B, )
        mse = err.square().mean(dim=(1, 2, 3)) # (B, )
        rmse = torch.sqrt(mse)
        psnr = 10 * torch.log10(1.0 / mse.clamp_min(eps))
        ssim = structural_similarity_index_measure(
            preds=preds,
            target=targets,
            data_range=1.0,
            reduction="none" # wait a sec to cal avg
        )

        mets = {k: v for k, v in zip(self.metrics.keys(), 
                                     [mae, mse, rmse, psnr, ssim])}

        for metric_name in self.metrics.keys():
            self.metrics[metric_name].update(value=mets[metric_name], weight=None)

    def compute(self):
        got = {name: metric.compute().item() for name, metric in self.metrics.items()}
        return got

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()



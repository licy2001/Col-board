import torch


def noise_estimation_loss(
    model,
    x0: torch.Tensor,
    x_cond: torch.Tensor,
    t: torch.LongTensor,
    e: torch.Tensor,
    b: torch.Tensor,
    keepdim=False,
):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x_cond, x], dim=1), t.float())
    pred_x0 = (x - output * (1.0 - a).sqrt()) / a.sqrt()
    if keepdim:
        loss = (e - output).square().sum(dim=(1, 2, 3))
    else:
        loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

    return loss, x, output, pred_x0


loss_registry = {
    "simple": noise_estimation_loss,
}

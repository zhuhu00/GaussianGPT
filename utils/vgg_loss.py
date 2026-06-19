import torch
import torchvision


class VggLoss(torch.nn.Module):
    def __init__(self, device, resize=True):
        super(VggLoss, self).__init__()
        blocks = []
        # split blocks after activation before pooling
        weights = torchvision.models.VGG19_Weights.DEFAULT
        blocks.append(torchvision.models.vgg19(weights=weights).features[:4].eval())
        blocks.append(torchvision.models.vgg19(weights=weights).features[4:9].eval())
        blocks.append(torchvision.models.vgg19(weights=weights).features[9:18].eval())
        blocks.append(torchvision.models.vgg19(weights=weights).features[18:27].eval())
        blocks.append(torchvision.models.vgg19(weights=weights).features[27:36].eval())

        self.loss_blocks = [0, 1, 2, 3, 4]  # blocks to contribute to the loss
        for bl in blocks:
            bl.to(device)
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the VGG loss between input and target images.
        Can take images with either 1 or 3 channels.
        If self.resize is True, it resizes the images to 224x224 before computing the loss.
        Args:
            input (torch.Tensor): Input image tensor of shape (N, C, H, W).
            target (torch.Tensor): Target image tensor of shape (N, C, H, W).
        """

        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        vgg_mean = [0.485, 0.456, 0.406]
        vgg_std = [0.229, 0.224, 0.225]
        input = torchvision.transforms.functional.normalize(
            input, mean=vgg_mean, std=vgg_std
        )
        target = torchvision.transforms.functional.normalize(
            target, mean=vgg_mean, std=vgg_std
        )
        if self.resize and (
            input.shape[2] != 224
            or input.shape[3] != 224
            or target.shape[2] != 224
            or target.shape[3] != 224
        ):
            input = self.transform(
                input, mode="bilinear", size=(224, 224), align_corners=False
            )
            target = self.transform(
                target, mode="bilinear", size=(224, 224), align_corners=False
            )
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in self.loss_blocks:
                loss += torch.nn.functional.mse_loss(x, y)
                # loss += torch.nn.functional.l1_loss(x, y)
                if i == self.loss_blocks[-1]:  # stop early
                    break
        return loss

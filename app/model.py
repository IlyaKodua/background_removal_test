import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from PIL import Image

class BiRefNetModel:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet", image_size: int = 1024):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = (image_size, image_size)

        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.to(self.device).eval()

        # half precision на GPU для скорости и экономии памяти
        if self.device == "cuda":
            self.model.half()
            torch.set_float32_matmul_precision("high")

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def remove_background(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        inputs = self.transform(image).unsqueeze(0).to(self.device)

        if self.device == "cuda":
            inputs = inputs.half()

        preds = self.model(inputs)[-1].sigmoid().float().cpu()
        pred = preds[0].squeeze()

        mask = transforms.ToPILImage()(pred).resize(image.size)
        result = image.copy()
        result.putalpha(mask)
        return result

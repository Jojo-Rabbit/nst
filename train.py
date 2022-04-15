import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from model import VGG


# Hyperparameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 365
TOTAL_STEPS = 200
LR = 0.02
ALPHA = 1e4
BETA = 1e-2

# Image Loader
loader = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                             transforms.ToTensor()])


def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = loader(img).unsqueeze(0)
    return img.to(DEVICE)


def train(model, optim, orig_img, gen_img, style_img):
    for _ in tqdm(range(TOTAL_STEPS)):
        orig_features = model(orig_img)
        gen_features = model(gen_img)
        style_features = model(style_img)

        content_loss = style_loss = 0

        for gen_feat, orig_feat, style_feat in zip(gen_features, orig_features, style_features):
            batch, channel, height, width = gen_feat.shape
            content_loss += torch.mean((gen_feat - orig_feat) ** 2)

            G = gen_feat.view(channel, height * width).mm(gen_feat.view(channel, height * width).t())
            A = style_feat.view(channel, height * width).mm(style_feat.view(channel, height * width).t())

            style_loss += torch.mean((G - A) ** 2)

        total_loss = ALPHA * content_loss + BETA * style_loss
        optim.zero_grad()
        total_loss.backward()
        optim.step()

    return [content_loss, style_loss, total_loss], gen_img


def main():
    original_image = load_image('Images/Input/Input_Timothee_1.jpg')
    style_image = load_image('Images/Style/Style_Picasso_1.jpg')
    generated_image = original_image.clone().requires_grad_(True)

    model = VGG().to(DEVICE).eval()
    optimizer = optim.Adam([generated_image], lr=LR, eps=1e-1)

    loss, output = train(model, optimizer, original_image, generated_image, style_image)

    print("Content Loss: ", loss[0])
    print("Style Loss: ", loss[1])
    print("Total Loss: ", loss[2])
    save_image(output, "Output/generated.jpg")


if __name__ == "__main__":
    main()
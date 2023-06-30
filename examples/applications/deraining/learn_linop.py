import os
import fire

import imageio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from derain.data import get_test_data
from derain.DGUNet_plus2 import DGUNet


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


@torch.no_grad()
def main(
    dataset='Rain100H',
    result_dir='results2/DGUNet_plus2',
    ckpt='checkpoints/DGUNet_plus2/model_best.pth',
    data_dir='datasets/test/',
):
    net = DGUNet().cuda().eval()
    load_checkpoint(net, ckpt)

    rgb_dir_test = os.path.join(data_dir, dataset, 'input')
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)

    result_dir = os.path.join(result_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)

    for data in tqdm(test_loader):
        input = data[0].cuda()
        filenames = data[1]

        output = net(input)
        output = torch.clamp(output[0], 0, 1)
        output = output.permute(0, 2, 3, 1).cpu().detach().numpy() * 255

        for batch in range(len(output)):
            restored_img = output[batch].astype('uint8')
            imageio.imsave(os.path.join(result_dir, filenames[batch] + '.png'),
                           restored_img)


if __name__ == '__main__':
    fire.Fire(main)

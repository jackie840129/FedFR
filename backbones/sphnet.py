import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, planes):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.prelu2 = nn.PReLU(planes)

    def forward(self, x):
        return x + self.prelu2(self.conv2(self.prelu1(self.conv1(x))))


class sphere(nn.Module):
    def __init__(self, type=20, is_gray=False,fp16=False):
        super(sphere, self).__init__()
        block = Block
        self.fp16 = fp16
        if type is 20:
            layers = [1, 2, 4, 1]
        elif type is 64:
            layers = [3, 7, 16, 3]
        else:
            raise ValueError('sphere' + str(type) + " IS NOT SUPPORTED! (sphere20 or sphere64)")
        filter_list = [3, 64, 128, 256, 512]
        if is_gray:
            filter_list[0] = 1

        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)
        self.fc = nn.Linear(512 * 7 * 7, 512)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    nn.init.normal_(m.weight, 0, 0.01)


    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, 3, stride, 1))
        layers.append(nn.PReLU(planes))
        for i in range(blocks):
            layers.append(block(planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = x.view(x.size(0), -1)

        x = self.fc(x.float() if self.fp16 else x)

        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

def sphnet(pretrained=False,dropout=None,fp16=False,type=64):
    return sphere(type,fp16=fp16)

if __name__ == '__main__':
    net = sphere(64)
    print(net)
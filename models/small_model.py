import torch
from torch import nn


def ConvBlock(first_chanels, second_chanels, kernel_size, dropout_rate):
    return nn.Sequential(
        nn.BatchNorm2d(first_chanels),
        nn.Conv2d(first_chanels, second_chanels, kernel_size, padding='same'),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.BatchNorm2d(second_chanels),
        nn.Conv2d(second_chanels, second_chanels, kernel_size, padding='same'),
        nn.ReLU(inplace=True)
    )


class Unet_Encoder(nn.Module):
    def __init__(self, kernel_size, dropout_rate, nkernels):
        super(Unet_Encoder, self).__init__()
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.nkernels = nkernels
        self.conv1 = ConvBlock(3, nkernels, self.kernel_size, self.dropout_rate)
        self.conv2 = ConvBlock(nkernels, nkernels*2, self.kernel_size, self.dropout_rate)
        self.conv3 = ConvBlock(nkernels*2, nkernels*4, self.kernel_size, self.dropout_rate)
        self.conv4 = ConvBlock(nkernels*4, nkernels*8, self.kernel_size, self.dropout_rate)
        self.maxpool_list = nn.ModuleList([nn.MaxPool2d(kernel_size=2) for _ in range(4)])
        self.conv_list = nn.ModuleList([self.conv1, self.conv2, self.conv3, self.conv4])
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

    def forward(self, input):
        list_skips = list()
        for i in range(4):
            skip = self.conv_list[i](input)
            input = self.maxpool_list[i](skip)
            list_skips.append(skip)
        return input, list_skips


class Unet_Decoder(nn.Module):
    def __init__(self, kernel_size, dropout_rate, nkernels):
        super(Unet_Decoder, self).__init__()
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.nkernels = nkernels
        self.conv5 = ConvBlock(nkernels*8, nkernels*16, self.kernel_size, self.dropout_rate)
        self.conv6 = ConvBlock(nkernels*16, nkernels*8, self.kernel_size, self.dropout_rate)
        self.conv7 = ConvBlock(nkernels*8, nkernels*4, self.kernel_size, self.dropout_rate)
        self.conv8 = ConvBlock(nkernels*4, nkernels*2, self.kernel_size, self.dropout_rate)
        self.conv_list = nn.ModuleList([self.conv5, self.conv6, self.conv7, self.conv8])
        self.convt_list = nn.ModuleList([nn.ConvTranspose2d(nkernels*(2**(4-i)), nkernels*((2**(4-i))//2), kernel_size=(2, 2), 
                                                            stride=(2, 2)) for i in range(4)])
        self.init_weights()

    
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)


    def forward(self, input, list_skips):
        for i in range(4):
            if i==0:
                out = self.conv_list[i](input)
                out = self.convt_list[i](out)
            else:
                out = self.conv_list[i](torch.cat((out, list_skips[4-i]), 1)) # channel
                out = self.convt_list[i](out)
        return out


class Model_Unet(nn.Module):
    def __init__(self, kernel_size, dropout_rate, nkernels, output_chanels):
        super(Model_Unet, self).__init__()
        self.output_chanels = output_chanels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.nkernels = nkernels
        self.enc_layer = Unet_Encoder(self.kernel_size, self.dropout_rate, self.nkernels)
        self.dec_layer = Unet_Decoder(self.kernel_size, self.dropout_rate, self.nkernels)
        self.conv9 = ConvBlock(self.nkernels*2, self.nkernels, self.kernel_size, self.dropout_rate)
        self.conv10 = nn.Conv2d(self.nkernels, self.output_chanels, (1, 1), padding='same')
        self.relu = nn.ReLU()
        self.activation = nn.Sigmoid()
        self.init_weights()
        

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

    
    def forward(self, input):
        out, list_skips = self.enc_layer(input)
        out = self.dec_layer(out, list_skips)
        out = self.conv9(torch.cat((out, list_skips[0]), 1)) # channel concat
        out = self.relu(out)
        out = self.conv10(out)
        #out = self.activation(out)                          # loss function with logit
        return out
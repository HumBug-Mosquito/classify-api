import math
import pickle

import framework.config as config
import skimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import LogmelFilterBank, Spectrogram


def move_data_to_gpu(x, cuda, using_float=False):
    if using_float:
        x = torch.Tensor(x)
    else:
        if 'float' in str(x.dtype):
            x = torch.Tensor(x)

        elif 'int' in str(x.dtype):
            x = torch.LongTensor(x)

        else:
            raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x

def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


# ----------------------------------------------------------------------------------------------------------------------

class MyLogmelFilterBank(LogmelFilterBank):
    def power_to_db(self, mel_spectrogram: torch.Tensor):
        # clamp to avoid log(0)
        amin = float(self.amin)
        ref  = float(self.ref)
        # make sure these are tensors in the same dtype + device
        amin_t = torch.as_tensor(amin, dtype=mel_spectrogram.dtype, device=mel_spectrogram.device)
        ref_t  = torch.as_tensor(ref,  dtype=mel_spectrogram.dtype, device=mel_spectrogram.device)

        # compute 10*log10(mel_spectrogram) entirely in PyTorch
        spec_db = 10.0 * torch.log10(torch.clamp(mel_spectrogram, min=amin_t))
        # subtract off the reference power
        spec_db = spec_db - (10.0 * torch.log10(ref_t))
        return spec_db
    
# ----------------------------------------------------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input):
        pool_size=(2, 2)
        pool_type='avg'
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_single_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):

        super(ConvBlock_single_layer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input):
        pool_size=(2, 2)
        pool_type='avg'
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_dilation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), dilation=(2,2), padding=(1, 1)):

        super(ConvBlock_dilation, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False, dilation=dilation)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False, dilation=dilation)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input):
        pool_size=(2, 2)
        pool_type='avg'
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_dilation_single_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), dilation=(2,2), padding=(1, 1)):

        super(ConvBlock_dilation_single_layer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False, dilation=dilation)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input,):
        pool_size=(2, 2)
        pool_type='avg'
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x



def load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def normalize(x, normalization_mel_file):
    # print('using: ', normalization_mel_file)
    norm_pickle = load_pickle(normalization_mel_file)
    mean = norm_pickle['mean']
    std = norm_pickle['std']
    return (x - mean) / std


class MTRCNN_pad_half(nn.Module):
    def __init__(self, class_num, dropout, MC_dropout, batchnormal, normalization_mel_file):

        super(MTRCNN_pad_half, self).__init__()

        norm_pickle = load_pickle(normalization_mel_file)
        self.register_buffer("mel_mean", torch.tensor(norm_pickle["mean"]).view(1, 1, 1, -1))
        self.register_buffer("mel_std",  torch.tensor(norm_pickle["std"]).view(1, 1, 1, -1))
        norm_pickle = None  # free memory

        # keep the file path only for the optional eager‑mode debug call
        self.normalization_mel_file = normalization_mel_file
        self.dropout = dropout
        self.MC_dropout = MC_dropout

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)

        frequency_num = 6
        frequency_emb_dim = 1
        # --------------------------------------------------------------------------------------------------------
        self.conv_block1 = ConvBlock_single_layer(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock_dilation_single_layer(in_channels=64, out_channels=128, padding=(1,1),
                                                           dilation=(2, 1))
        self.conv_block3 = ConvBlock_dilation_single_layer(in_channels=128, out_channels=256, padding=(1,1),
                                                           dilation=(3, 1))
        self.k_3_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # -------------- kernel 5
        kernel_size = (5, 5)
        self.conv_block1_kernel_5 = ConvBlock_single_layer(in_channels=1, out_channels=64, kernel_size=kernel_size,
                                                           padding=(2,2))
        self.conv_block2_kernel_5 = ConvBlock_dilation_single_layer(in_channels=64, out_channels=128,
                                                                    kernel_size=kernel_size,
                                                       padding=(2, 2), dilation=(2, 1))
        self.conv_block3_kernel_5 = ConvBlock_dilation_single_layer(in_channels=128, out_channels=256,
                                                                    kernel_size=kernel_size,
                                                       padding=(2, 2), dilation=(3, 1))
        self.k_5_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        # -------------- kernel 7
        kernel_size = (7, 7)
        self.conv_block1_kernel_7 = ConvBlock_single_layer(in_channels=1, out_channels=64, kernel_size=kernel_size,
                                                           padding=(3, 3))
        self.conv_block2_kernel_7 = ConvBlock_dilation_single_layer(in_channels=64, out_channels=128,
                                                                    kernel_size=kernel_size,
                                                       padding=(3, 3), dilation=(2, 1))
        self.conv_block3_kernel_7 = ConvBlock_dilation_single_layer(in_channels=128, out_channels=256, kernel_size=kernel_size,
                                                       padding=(3, 3), dilation=(3, 1))
        self.k_7_freq_to_1 = nn.Linear(frequency_num, frequency_emb_dim, bias=True)

        ##################################### gnn ####################################################################


        # ----------------------------------------------------------------------------------------------------


        scene_event_embedding_dim = 512
        # embedding layers
        self.fc_embedding_event = nn.Linear(256*3, scene_event_embedding_dim, bias=True)
        # -----------------------------------------------------------------------------------------------------------

        self.fc_final = nn.Linear(scene_event_embedding_dim, class_num, bias=True)

        ##############################################################################################################
        mel_bins = 64
        sample_rate = 8000  # 16000
        fmax = int(sample_rate / 2)
        fmin = 50
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        # window_size = 512  # 16Khz
        # hop_size = 160  # 16Khz
        window_size = 256 # int(np.floor(1024 * (sample_rate / 32000)))
        hop_size = 80# int(np.floor(320 * (sample_rate / 32000)))  # 10ms
        # print(window_size, hop_size)  # 256 80

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = MyLogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)
        
        # ensure the mel‐filter buffer is default‐contiguous
        for name, buf in self.logmel_extractor.named_buffers():
            # this will replace the buffer in‐place with a contiguous copy
            self.logmel_extractor.register_buffer(name, buf.contiguous())
            
        for name, param in self.logmel_extractor.named_parameters():
            param.data = param.data.contiguous()
            
        self.normalization_mel_file = normalization_mel_file
        self.init_weight()

    def init_weight(self):
        if self.batchnormal:
            init_bn(self.bn0)

        init_layer(self.fc_embedding_event)


    def mean_max(self, x):
           # 1) do both reductions with keepdim=True so XNNPACK can lower them
        x1, _ = torch.max(x, dim=2, keepdim=True)   # shape: (B, C, 1, M)
        x2     = torch.mean(x,   dim=2, keepdim=True)  # shape: (B, C, 1, M)

        # 2) combine
        y = x1 + x2                                 # still (B, C, 1, M)

        # 3) drop that singleton time‐axis if later code expects (B, C, M)
        return y.squeeze(2)
    
    def maybe_dropout(self, t: torch.Tensor) -> torch.Tensor:
        # TorchScript‑friendly helper
        if self.MC_dropout and self.dropout > 0.0:
            return F.dropout(t, p=self.dropout, training=self.training)
        return t
    

    def torch_view_as_windows(self, feat: torch.Tensor, win_size: int, step: int):
        """
        Sliding windows over a (time_steps × mel_bins) tensor.
        Returns: Tensor of shape (num_windows, 1, win_size, mel_bins).
        """
        # 1) reshape → (N=1, C=1, H=time_steps, W=mel_bins)
        x = feat.unsqueeze(0).unsqueeze(0)

        # 2) unfold with kernel=(win_size, mel_bins) sliding along H only
        #    stride=(step, mel_bins) so we move 'step' in H each window
        cols = F.unfold(
            x,
            kernel_size=(win_size, feat.size(1)),
            stride=(step, feat.size(1))
        )
        # cols: shape (1, win_size*mel_bins, num_windows)

        # 3) collapse batch-dim and transpose → (num_windows, win_size*mel_bins)
        cols = cols[0].transpose(0, 1).contiguous()

        # 4) reshape → (num_windows, win_size, mel_bins)
        num_windows = cols.size(0)
        windows = cols.view(num_windows, win_size, feat.size(1))

        # 5) add channel dim → (num_windows, 1, win_size, mel_bins)
        return windows.unsqueeze(1)
    

    def forward(self, wav):
        """
        Input: pcm data
        """
    

        x = self.spectrogram_extractor(wav)  # (batch_size, 1, time_steps, freq_bins)
        x = x.contiguous()       
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        x = x.contiguous()       
        
        feat = x[0, 0]
        # feat = feat.T  # (mel_bins, time_steps)
        
        # feats_windowed = skimage.util.view_as_windows(feat.T, (config.win_size, config.mel_bins), step=config.step_size)
        # print(feats_windowed.shape)
        
        feats_windowed = self.torch_view_as_windows(
            feat, win_size=config.win_size, step=config.step_size
        )
        
        # print('feats_windowed: ', feats_windowed.shape)  # cpu (32, 1, 200, 64)
        
        windows = feats_windowed[:, 0].contiguous()    # (nW, win, mel)
        mean    = self.mel_mean.squeeze(0).squeeze(0).squeeze(0)  # (mel,)
        std     = self.mel_std .squeeze(0).squeeze(0).squeeze(0)  # (mel,)

        # Now both windows and mean/std are 3D contiguous:
        x = ((windows - mean) / std).contiguous()     # -> all_contiguous == True
        # x =     ((feats_windowed[:, 0] - self.mel_mean) / self.mel_std).squeeze(0)
        # feats_windowed: (nW,1,win,mel)
        # strip channel dim and normalize via the buffers directly:
        # x = (feats_windowed[:,0] - self.mel_mean) / self.mel_std
        x = move_data_to_gpu(x, cuda=False)[:, None]
        
        if self.batchnormal:
            x = x.transpose(1, 3).contiguous()
            x = self.bn0(x)
            x = x.transpose(1, 3).contiguous()

        batch_x = x

        # print(x.size())  #  torch.Size([64, 1, 100, 64])
        x_k_3 = self.conv_block1(batch_x)
        x_k_3 = self.maybe_dropout(x_k_3)

        x_k_3 = self.conv_block2(x_k_3)
        x_k_3 = self.maybe_dropout(x_k_3)
        x_k_3 = self.conv_block3(x_k_3)
        x_k_3 = self.maybe_dropout(x_k_3)
        # print('x_k_3: ', x_k_3.size())  # x_k_3:  torch.Size([64, 64, 8, 6])

        x_k_3 = self.mean_max(x_k_3)
        # print('x_k_3: ', x_k_3.size())  # x_k_3:  torch.Size([32, 64, 1])
        x_k_3_mel = F.relu_(x_k_3[:, :, 0])
        # print('x_k_3_mel: ', x_k_3_mel.size())  # x_k_3_mel:  torch.Size([64, 64])

        # kernel 5 -----------------------------------------------------------------------------------------------------
        x_k_5 = self.conv_block1_kernel_5(batch_x)
        x_k_5 = self.maybe_dropout(x_k_5)
        # print(x_k_5.size())  # torch.Size([64, 16, 48, 32])

        x_k_5 = self.conv_block2_kernel_5(x_k_5)
        x_k_5 = self.maybe_dropout(x_k_5)
        # print(x_k_5.size())  # torch.Size([64, 32, 20, 15])

        x_k_5 = self.conv_block3_kernel_5(x_k_5)
        x_k_5 = self.maybe_dropout(x_k_5)
        # print(x_k_5.size(), '\n')  # torch.Size([64, 64, 4, 6])

        x_k_5 = self.mean_max(x_k_5)  # torch.Size([32, 64, 1])
        x_k_5_mel = F.relu_(x_k_5[:, :, 0])
        # print('x_k_5_mel: ', x_k_5_mel.size())  # torch.Size([64, 64])

        # kernel 7 -----------------------------------------------------------------------------------------------------
        x_k_7 = self.conv_block1_kernel_7(batch_x)
        x_k_7 = self.maybe_dropout(x_k_7)
        # print(x_k_7.size())  # torch.Size([64, 16, 47, 32])

        x_k_7 = self.conv_block2_kernel_7(x_k_7)
        x_k_7 = self.maybe_dropout(x_k_7)
        # print(x_k_7.size())  # torch.Size([64, 32, 17, 15])

        x_k_7 = self.conv_block3_kernel_7(x_k_7)
        x_k_7 = self.maybe_dropout(x_k_7)
        # print(x_k_7.size(), '\n')  # torch.Size([64, 64, 2, 6])

        x_k_7 = self.mean_max(x_k_7)
        # print(x_k_7.size(), '\n')  # torch.Size([32, 64, 1])
        x_k_7_mel = F.relu_(x_k_7[:, :, 0])
        # print('x_k_7_mel: ', x_k_7_mel.size())  #torch.Size([64, 64])

        # -------------------------------------------------------------------------------------------------------------
        event_embs_log_mel = torch.cat([x_k_3_mel, x_k_5_mel,
                                        x_k_7_mel], dim=-1)
        # print(event_embs_log_mel.size())  # torch.Size([64, 192])  (node_num, batch, edge_dim)

        # -------------------------------------------------------------------------------------------------------------
        event_embeddings = F.relu_(self.fc_embedding_event(event_embs_log_mel))
        # -------------------------------------------------------------------------------------------------------------

        event = self.fc_final(self.maybe_dropout(event_embeddings))

        event = torch.sigmoid(event)

        return event


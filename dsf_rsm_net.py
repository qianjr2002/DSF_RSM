import torch
import torch.nn as nn
import torch.nn.functional as F

from mc_stft import multi_channeled_STFT

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution layer for efficient spatial filtering.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                  groups=in_channels, padding=kernel_size//2)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DSFModule(nn.Module):
    """
    Dynamic Spatial Filtering (DSF) module that performs adaptive beamforming.
    """
    def __init__(self, num_channels, num_freq_bins, kernel_size=3):
        super().__init__()
        self.num_channels = num_channels
        self.num_freq_bins = num_freq_bins
        
        # Number of unique IPD pairs
        self.num_ipd_pairs = num_channels * (num_channels - 1) // 2
        
        # Input processing (magnitude + IPD)
        self.dsf_conv = nn.Sequential(
            DepthwiseSeparableConv(1 + self.num_ipd_pairs, 64, kernel_size),
            nn.ReLU(),
            DepthwiseSeparableConv(64, 32, kernel_size),
            nn.ReLU(),
            nn.Conv2d(32, num_channels, kernel_size, padding=kernel_size//2)
        )
        
    def compute_ipd(self, x_phase):
        """
        Compute inter-channel phase differences (IPDs)
        x_phase: [batch, channels, freq_bins, time_frames]
        returns: [batch, ipd_pairs, freq_bins, time_frames]
        """
        batch, channels, freq_bins, time_frames = x_phase.shape
        ipd_pairs = []
        
        # Compute IPD for each unique pair of channels
        for i in range(channels):
            for j in range(i+1, channels):
                ipd_pairs.append(x_phase[:, i] - x_phase[:, j])
        
        ipd = torch.stack(ipd_pairs, dim=1)  # [batch, ipd_pairs, freq_bins, time_frames]
        return ipd
    
    def forward(self, x_complex):
        """
        x_complex: complex input tensor [batch, channels, freq_bins, time_frames]
        returns: beamformed complex output [batch, freq_bins, time_frames]
        """
        # Extract magnitude and phase
        x_mag = torch.abs(x_complex)  # [batch, channels, freq_bins, time_frames]
        x_phase = torch.angle(x_complex)  # [batch, channels, freq_bins, time_frames]
        
        # Compute IPD features
        ipd = self.compute_ipd(x_phase)  # [batch, ipd_pairs, freq_bins, time_frames]
        
        # Use mean magnitude across channels as input feature
        mean_mag = torch.mean(x_mag, dim=1, keepdim=True)  # [batch, 1, freq_bins, time_frames]
        
        # Concatenate magnitude and IPD features
        features = torch.cat([mean_mag, ipd], dim=1)  # [batch, 1+ipd_pairs, freq_bins, time_frames]
        
        # Generate spatial filters
        filters = self.dsf_conv(features)  # [batch, channels, freq_bins, time_frames]
        
        # Apply spatial filters to input
        # We need to handle complex multiplication
        beamformed = torch.sum(filters * x_complex, dim=1)  # [batch, freq_bins, time_frames]
        
        return beamformed

class RSMNetwork(nn.Module):
    """
    Residual Spectral Mapping (RSM) network that refines the beamformed output.
    """
    def __init__(self, num_freq_bins, kernel_size=3):
        super().__init__()
        
        self.residual_net = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(2)
        )
        
    def forward(self, beamformed_complex):
        """
        beamformed_complex: complex input tensor [batch, freq_bins, time_frames]
        returns: enhanced complex output [batch, freq_bins, time_frames]
        """
        # Split into real and imaginary parts
        if torch.is_complex(beamformed_complex):
            imag = beamformed_complex.imag.unsqueeze(1)
            real = beamformed_complex.real.unsqueeze(1)
        else:
            # 模拟输入情况，仅用于模型复杂度估计
            B, F, T = beamformed_complex.shape
            imag = torch.zeros((B, 1, F, T), dtype=beamformed_complex.dtype, device=beamformed_complex.device)
            real = torch.zeros((B, 1, F, T), dtype=beamformed_complex.dtype, device=beamformed_complex.device)
        
        # Concatenate along channel dimension
        x = torch.cat([real, imag], dim=1)  # [batch, 2, freq_bins, time_frames]
        
        # Predict residuals
        residuals = self.residual_net(x)  # [batch, 2, freq_bins, time_frames]
        
        # Split residuals
        res_real = residuals[:, 0]  # [batch, freq_bins, time_frames]
        res_imag = residuals[:, 1]  # [batch, freq_bins, time_frames]
        
        # Add residuals to beamformed output
        enhanced_real = real.squeeze(1) + res_real
        enhanced_imag = imag.squeeze(1) + res_imag
        
        # Combine back to complex tensor
        enhanced = torch.complex(enhanced_real, enhanced_imag)
        
        return enhanced

class DSF_RSM(nn.Module):
    """
    Complete DSF-RSM model combining both modules.
    """
    def __init__(self, num_channels, num_freq_bins):
        super().__init__()
        self.dsf = DSFModule(num_channels, num_freq_bins)
        self.rsm = RSMNetwork(num_freq_bins)
        
    def forward(self, x_complex):
        """
        x_complex: complex input tensor [batch, channels, freq_bins, time_frames]
        returns: enhanced complex output [batch, freq_bins, time_frames]
        """
        # Apply dynamic spatial filtering
        beamformed = self.dsf(x_complex)
        
        # Apply residual spectral mapping
        enhanced = self.rsm(beamformed)       
        return enhanced




def test_dsf_rsm():
    wav = torch.rand(B, C, T)
    print(f"Input shape: {wav.shape}")
    stft_module = multi_channeled_STFT(n_fft=n_fft, hop_length=hop_length)
    # STFT: [B, C, F, T']
    spec = stft_module.to_spec_complex(wav)
    print(f"model input shape: {spec.shape} spec.dtype {spec.dtype}") # model input shape: torch.Size([16, 2, 257, 63]) spec.dtype torch.complex64
    model = DSF_RSM(num_channels=C, num_freq_bins=num_freq_bins)
    est_spec = model(spec)
    print(f"model output shape: {est_spec.shape}") # model output shape: torch.Size([16, 257, 63])
    # B, F, T_ = est_spec.shape
    est_wav = torch.istft(
        est_spec,
        n_fft=512,
        hop_length=256,
        win_length=512,
        length=wav.shape[-1]
    )  # [B, T]
    est_wav = est_wav.unsqueeze(1)  # [B, 1, T]   
    print(f"Output shape: {est_wav.shape}") # Output shape: torch.Size([16, 1, 16000])


def test_model_complexity_info():
    from ptflops import get_model_complexity_info
    model = DSF_RSM(num_channels=C, num_freq_bins=num_freq_bins)
    flops, params = get_model_complexity_info(model,
                                              (2, 257, 63),
                                              as_strings=True,
                                              print_per_layer_stat=False)
    print(f'flops:{flops}, params:{params}')
    # flops:704.57 MMac, params:43.07 k


if __name__ == "__main__":
    B, C, T = 16, 2, 16000
    n_fft, hop_length =512, 256
    num_freq_bins = n_fft//2 + 1

    test_dsf_rsm()
    test_model_complexity_info()
import torch
import torch.nn.functional as F

class StegoImpactAttention():
    def get_rho(self, cover):
        # cover: (batch_size, 3, H, W)
        batch_size, channels, H, W = cover.shape
        
        # Define Daubechies 8 filters
        hpdf = torch.tensor([-0.0544158422, 0.3128715909, -0.6756307363,
                             0.5853546837, 0.0158291053, -0.2840155430,
                             -0.0004724846, 0.1287474266, 0.0173693010,
                             -0.0440882539, -0.0139810279, 0.0087460940,
                             0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768], device=cover.device, dtype=cover.dtype)
        
        lpdf = (-1) ** torch.arange(len(hpdf), device=cover.device) * hpdf.flip(0)
        
        # Create 2D filters
        F_filters = []
        for i in range(3):
            if i == 0:
                F_filters.append(torch.ger(lpdf, hpdf))
            elif i == 1:
                F_filters.append(torch.ger(hpdf, lpdf))
            else:
                F_filters.append(torch.ger(hpdf, hpdf))
        
        # Calculate padding size
        pad_size = max([f.shape[0] for f in F_filters]) -1
        
        # Apply padding
        cover_padded = F.pad(cover, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        # Calculate xi
        xi = []
        for F_filter in F_filters:
            F_filter = F_filter.unsqueeze(0).unsqueeze(0).to(cover.device)  # (1, 1, k, k)
            F_filter_rot = F_filter.flip(-2, -1)
            
            R = F.conv2d(cover_padded, F_filter.expand(channels, -1, -1, -1), groups=channels)
            abs_R = torch.abs(R)
            
            xi_i = F.conv2d(abs_R, torch.abs(F_filter_rot).expand(channels, -1, -1, -1), groups=channels)
            xi.append(xi_i)
        
        # Calculate rho
        xi_stack = torch.stack(xi, dim=0)  # (3, batch_size, channels, H, W)
        p = -1
        rho = (xi_stack ** p).sum(dim=0) ** (-1 / p)
        
        # Thresholding and NaN handling
        wetCost = 1e10
        rho = torch.clamp(rho, max=wetCost)
        rho[torch.isnan(rho)] = wetCost

        # Normalize rho
        min_rho = rho.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_rho = rho.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        rho = (rho - min_rho) / (max_rho - min_rho + 1e-8)
        
        # Invert rho for attention
        rho = 1 - rho  
        
        return rho
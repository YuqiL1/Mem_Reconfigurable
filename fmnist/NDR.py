import torch

class NDRActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, xs, ys, alpha, beta,
                x_min_neg, slope_neg, intercept_neg,
                x_max_mid, slope_pos):
        ctx.u_shape = u.shape
        ctx.x_min_neg = x_min_neg
        ctx.slope_neg = slope_neg
        ctx.intercept_neg = intercept_neg
        ctx.x_max_mid = x_max_mid
        ctx.slope_pos = slope_pos

        u_flat = u.reshape(-1)
        N = xs.numel()
                  
        mask_neg = u_flat < x_min_neg
        mask_mid = (u_flat >= x_min_neg) & (u_flat <= x_max_mid)
        mask_pos = u_flat > x_max_mid

        y_neg = slope_neg * u_flat - intercept_neg

        idx = torch.searchsorted(xs, u_flat, right=False)
        i0 = torch.clamp(idx - 1, 0, N-1)
        i1 = torch.clamp(idx, 0, N-1)
        x0, x1 = xs[i0], xs[i1]
        y0, y1 = ys[i0], ys[i1]
        slope_mid = (y1 - y0) / (x1 - x0 + 1e-8)
        y_mid = y0 + slope_mid * (u_flat - x0)

        y_pos = slope_pos * u_flat

        y_final = torch.empty_like(y_neg)
        y_final[mask_neg] = y_neg[mask_neg]
        y_final[mask_mid] = y_mid[mask_mid]
        y_final[mask_pos] = y_pos[mask_pos]

        y_scaled = alpha * y_final + beta

        ctx.save_for_backward(mask_neg, mask_mid, mask_pos, slope_mid, alpha, y_final)

        return y_scaled.reshape(ctx.u_shape)

    @staticmethod
    def backward(ctx, grad_output):
        mask_neg, mask_mid, mask_pos, slope_mid, alpha, y_final = ctx.saved_tensors
        slope_neg = ctx.slope_neg
        slope_pos = ctx.slope_pos

        grad_flat = grad_output.reshape(-1)
        grad_y = grad_flat * alpha

        slope_comb = torch.where(mask_neg, slope_neg,
                       torch.where(mask_mid, slope_mid, slope_pos))

        grad_u_flat = grad_y * slope_comb
        grad_u = grad_u_flat.reshape(ctx.u_shape)

        grad_alpha = (grad_flat * y_final).sum().unsqueeze(0)
        grad_beta = grad_flat.sum().unsqueeze(0)

        return grad_u, None, None, grad_alpha, grad_beta, None, None, None, None, None

class NDRActivation(torch.nn.Module):
    def __init__(self, excel_path: str, col_x, col_y,
                 sheet_name=0, skiprows=None, usecols=None):
        super().__init__()
        import pandas as pd
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        df = pd.read_excel(excel_path,
                           sheet_name=sheet_name,
                           skiprows=skiprows,
                           usecols=usecols,
                           engine='openpyxl')
        xs_raw = df.iloc[:, col_x].values if isinstance(col_x, int) else df[col_x].values
        ys_raw = df.iloc[:, col_y].values if isinstance(col_y, int) else df[col_y].values
        xs = torch.tensor(-xs_raw, dtype=torch.float32)
        ys = torch.tensor(-ys_raw, dtype=torch.float32)
        xs, idx = torch.sort(xs)
        ys = ys[idx]

        self.register_buffer('xs', xs)
        self.register_buffer('ys', ys)

        self.register_buffer('x_min_neg', torch.tensor(-0.000408, dtype=torch.float32))
        self.register_buffer('slope_neg', torch.tensor(532.0345, dtype=torch.float32))
        self.register_buffer('intercept_neg', torch.tensor(0.401671, dtype=torch.float32))
        self.register_buffer('x_max_mid', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('slope_pos', torch.tensor(20000.0, dtype=torch.float32))

        self.in_alpha = torch.nn.Parameter(torch.tensor(0.01))
        self.in_beta  = torch.nn.Parameter(torch.tensor(0.0))
        self.alpha    = torch.nn.Parameter(torch.tensor(1.0))
        self.beta     = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        u = self.in_alpha * x + self.in_beta
        return NDRActivationFunction.apply(
            u,
            self.xs,
            self.ys,
            self.alpha,
            self.beta,
            self.x_min_neg,
            self.slope_neg,
            self.intercept_neg,
            self.x_max_mid,
            self.slope_pos
        )


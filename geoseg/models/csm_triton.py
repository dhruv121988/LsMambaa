import torch
import warnings

WITH_TRITON = True
# WITH_TRITON = False
try:
    import triton
    import triton.language as tl
except:
    WITH_TRITON = False
    warnings.warn("Triton not installed, fall back to pytorch implements.")

# to make sure cached_property can be loaded for triton
if WITH_TRITON:
    try:
        from functools import cached_property
    except:
        warnings.warn("if you are using py37, add this line to functools.py: "
            "cached_property = lambda func: property(lru_cache()(func))")

# torch implementation ========================================
def cross_scan_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, C, H, W = x.shape
        if scans == 0:
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)
            y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])
        elif scans == 1:
            y = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        elif scans == 2:
            y = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
            y = torch.cat([y, y.flip(dims=[-1])], dim=1)
    else:
        B, H, W, C = x.shape
        if scans == 0:
            y = x.new_empty((B, H * W, 4, C))
            y[:, :, 0, :] = x.flatten(1, 2)
            y[:, :, 1, :] = x.transpose(dim0=1, dim1=2).flatten(1, 2)
            y[:, :, 2:4, :] = torch.flip(y[:, :, 0:2, :], dims=[1])
        elif scans == 1:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 4, 1)
        elif scans == 2:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 2, 1)
            y = torch.cat([y, y.flip(dims=[1])], dim=2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        if scans == 0:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y[:, 0] + y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        elif scans == 1:
            y = y.sum(1)
        elif scans == 2:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y.sum(1)
    else:
        B, H, W, K, D = y.shape
        y = y.view(B, -1, K, D)
        if scans == 0:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y[:, :, 0] + y[:, :, 1].view(B, W, H, -1).transpose(dim0=1, dim1=2).contiguous().view(B, -1, D)        
        elif scans == 1:
            y = y.sum(2)
        elif scans == 2:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y.sum(2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()
    
    return y


def cross_scan1b1_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, _, C, H, W = x.shape
        if scans == 0:
            y = torch.stack([
                x[:, 0].flatten(2, 3),
                x[:, 1].transpose(dim0=2, dim1=3).flatten(2, 3),
                torch.flip(x[:, 2].flatten(2, 3), dims=[-1]),
                torch.flip(x[:, 3].transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]),
            ], dim=1)
        elif scans == 1:
            y = x.flatten(2, 3)
        elif scans == 2:
            y = torch.stack([
                x[:, 0].flatten(2, 3),
                x[:, 1].flatten(2, 3),
                torch.flip(x[:, 2].flatten(2, 3), dims=[-1]),
                torch.flip(x[:, 3].flatten(2, 3), dims=[-1]),
            ], dim=1)
    else:
        B, H, W, _, C = x.shape
        if scans == 0:
            y = torch.stack([
                x[:, :, :, 0].flatten(1, 2),
                x[:, :, :, 1].transpose(dim0=1, dim1=2).flatten(1, 2),
                torch.flip(x[:, :, :, 2].flatten(1, 2), dims=[1]),
                torch.flip(x[:, :, :, 3].transpose(dim0=1, dim1=2).flatten(1, 2), dims=[1]),
            ], dim=2)
        elif scans == 1:
            y = x.flatten(1, 2)
        elif scans == 2:
            y = torch.stack([
                x[:, 0].flatten(1, 2),
                x[:, 1].flatten(1, 2),
                torch.flip(x[:, 2].flatten(1, 2), dims=[-1]),
                torch.flip(x[:, 3].flatten(1, 2), dims=[-1]),
            ], dim=2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge1b1_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        if scans == 0:
            y = torch.stack([
                y[:, 0],
                y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).flatten(2, 3),
                torch.flip(y[:, 2], dims=[-1]),
                torch.flip(y[:, 3].view(B, -1, W, H).transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]),
            ], dim=1)
        elif scans == 1:
            y = y
        elif scans == 2:
            y = torch.stack([
                y[:, 0],
                y[:, 1],
                torch.flip(y[:, 2], dims=[-1]),
                torch.flip(y[:, 3], dims=[-1]),
            ], dim=1)
    else:
        B, H, W, K, D = y.shape
        y = y.view(B, -1, K, D)
        if scans == 0:
            y = torch.stack([
                y[:, :, 0],
                y[:, :, 1].view(B, W, H, -1).transpose(dim0=1, dim1=2).flatten(1, 2),
                torch.flip(y[:, :, 2], dims=[1]),
                torch.flip(y[:, :, 3].view(B, W, H, -1).transpose(dim0=1, dim1=2).flatten(1, 2), dims=[1]),
            ], dim=2)
        elif scans == 1:
            y = y
        elif scans == 2:
            y = torch.stack([
                y[:, :, 0],
                y[:, :, 1],
                torch.flip(y[:, :, 2], dims=[1]),
                torch.flip(y[:, :, 3], dims=[1]),
            ], dim=2)

    if out_channel_first and (not in_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not out_channel_first) and in_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


class CrossScanF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        if one_by_one:
            B, K, C, H, W = x.shape
            if not in_channel_first:
                B, H, W, K, C = x.shape
        else:
            B, C, H, W = x.shape
            if not in_channel_first:
                B, H, W, C = x.shape
        ctx.shape = (B, C, H, W)

        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        y = _fn(x, in_channel_first, out_channel_first, scans)

        return y
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape

        ys = ys.view(B, -1, C, H, W) if out_channel_first else ys.view(B, H, W, -1, C)
        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        y = _fn(ys, in_channel_first, out_channel_first, scans)
        
        if one_by_one:
            y = y.view(B, 4, -1, H, W) if in_channel_first else y.view(B, H, W, 4, -1)
        else:
            y = y.view(B, -1, H, W) if in_channel_first else y.view(B, H, W, -1)

        return y, None, None, None, None


class CrossMergeF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        B, K, C, H, W = ys.shape
        if not out_channel_first:
            B, H, W, K, C = ys.shape
        ctx.shape = (B, C, H, W)
        
        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        y = _fn(ys, in_channel_first, out_channel_first, scans)

        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, h, w)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
    
        if not one_by_one:
            if in_channel_first:
                x = x.view(B, C, H, W)
            else:
                x = x.view(B, H, W, C)
        else:
            if in_channel_first:
                x = x.view(B, 4, C, H, W)
            else:
                x = x.view(B, H, W, 4, C)   
                     
        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        x = _fn(x, in_channel_first, out_channel_first, scans)
        x = x.view(B, 4, C, H, W) if out_channel_first else x.view(B, H, W, 4, C)
    
        return x, None, None, None, None


# triton implements ========================================

@triton.jit
def triton_cross_scan_flex(
    x: tl.tensor, # (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
    y: tl.tensor, # (B, 4, C, H, W) | (B, H, W, 4, C)
    x_layout: tl.constexpr,
    y_layout: tl.constexpr,
    operation: tl.constexpr,
    onebyone: tl.constexpr,
    scans: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    # x_layout = 0
    # y_layout = 1 # 0 BCHW, 1 BHWC
    # operation = 0 # 0 scan, 1 merge
    # onebyone = 0 # 0 false, 1 true
    # scans = 0 # 0 cross scan, 1 unidirectional, 2 bidirectional

    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    pos_h = (i_h * BH + tl.arange(0, BH)[:, None])
    pos_w = (i_w * BW + tl.arange(0, BW)[None, :])
    neg_h = (DH - i_h * BH - 1 - tl.arange(0, BH)[:, None])
    neg_w = (DW - i_w * BW - 1 - tl.arange(0, BW)[None, :])
    if scans == 0:
        # none; trans; flip; trans + flip;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = pos_w * DH + pos_h # trans
        HWRoute2 = neg_h * DW + neg_w # flip
        HWRoute3 = neg_w * DH + neg_h # trans + flip
    elif scans == 1:
        # none; none; none; none;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = HWRoute0
        HWRoute2 = HWRoute0
        HWRoute3 = HWRoute0
    elif scans == 2:
        # none; none; flip; flip;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = HWRoute0
        HWRoute2 = neg_h * DW + neg_w # flip
        HWRoute3 = HWRoute2      

    _tmp1 = DC * DH * DW

    y_ptr_base = y + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if y_layout == 0 else i_c * BC)
    if y_layout == 0:
        p_y1 = y_ptr_base + HWRoute0
        p_y2 = y_ptr_base + _tmp1 + HWRoute1
        p_y3 = y_ptr_base + 2 * _tmp1 + HWRoute2
        p_y4 = y_ptr_base + 3 * _tmp1 + HWRoute3
    else:
        p_y1 = y_ptr_base + HWRoute0 * 4 * DC
        p_y2 = y_ptr_base + DC + HWRoute1 * 4 * DC
        p_y3 = y_ptr_base + 2 * DC + HWRoute2 * 4 * DC
        p_y4 = y_ptr_base + 3 * DC + HWRoute3 * 4 * DC       
    
    if onebyone == 0:
        x_ptr_base = x + i_b * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
        if x_layout == 0:
            p_x = x_ptr_base + HWRoute0
        else:
            p_x = x_ptr_base + HWRoute0 * DC

        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _x = tl.load(p_x + _idx_x, mask=_mask_hw)
                tl.store(p_y1 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y2 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y3 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y4 + _idx_y, _x, mask=_mask_hw)
        elif operation == 1:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _y1 = tl.load(p_y1 + _idx_y, mask=_mask_hw)
                _y2 = tl.load(p_y2 + _idx_y, mask=_mask_hw)
                _y3 = tl.load(p_y3 + _idx_y, mask=_mask_hw)
                _y4 = tl.load(p_y4 + _idx_y, mask=_mask_hw)
                tl.store(p_x + _idx_x, _y1 + _y2 + _y3 + _y4, mask=_mask_hw)

    else:
        x_ptr_base = x + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
        if x_layout == 0:
            p_x1 = x_ptr_base + HWRoute0
            p_x2 = p_x1 + _tmp1
            p_x3 = p_x2 + _tmp1
            p_x4 = p_x3 + _tmp1  
        else:
            p_x1 = x_ptr_base + HWRoute0 * 4 * DC
            p_x2 = p_x1 + DC
            p_x3 = p_x2 + DC
            p_x4 = p_x3 + DC        
    
        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                tl.store(p_y1 + _idx_y, tl.load(p_x1 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y2 + _idx_y, tl.load(p_x2 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y3 + _idx_y, tl.load(p_x3 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y4 + _idx_y, tl.load(p_x4 + _idx_x, mask=_mask_hw), mask=_mask_hw)
        else:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                tl.store(p_x1 + _idx_x, tl.load(p_y1 + _idx_y), mask=_mask_hw)
                tl.store(p_x2 + _idx_x, tl.load(p_y2 + _idx_y), mask=_mask_hw)
                tl.store(p_x3 + _idx_x, tl.load(p_y3 + _idx_y), mask=_mask_hw)
                tl.store(p_x4 + _idx_x, tl.load(p_y4 + _idx_y), mask=_mask_hw)


class CrossScanTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        if one_by_one:
            if in_channel_first:
                B, _, C, H, W = x.shape
            else:
                B, H, W, _, C = x.shape
        else:
            if in_channel_first:
                B, C, H, W = x.shape
            else:
                B, H, W, C = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = 1, 32, 32
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)

        y = x.new_empty((B, 4, C, H * W)) if out_channel_first else x.new_empty((B, H * W, 4, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x.contiguous(), y, 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans, 
            BC, BH, BW, C, H, W, NH, NW
        )
        return y
        
    @staticmethod
    def backward(ctx, y: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        if one_by_one:
            x = y.new_empty((B, 4, C, H, W)) if in_channel_first else y.new_empty((B, H, W, 4, C))
        else:
            x = y.new_empty((B, C, H, W)) if in_channel_first else y.new_empty((B, H, W, C))
        
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x, y.contiguous(), 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return x, None, None, None, None


class CrossMergeTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        if out_channel_first:
            B, _, C, H, W = y.shape
        else:
            B, H, W, _, C = y.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = 1, 32, 32
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        if one_by_one:
            x = y.new_empty((B, 4, C, H * W)) if in_channel_first else y.new_empty((B, H * W, 4, C))
        else:
            x = y.new_empty((B, C, H * W)) if in_channel_first else y.new_empty((B, H * W, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x, y.contiguous(), 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return x
        
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        y = x.new_empty((B, 4, C, H, W)) if out_channel_first else x.new_empty((B, H, W, 4, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x.contiguous(), y, 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return y, None, None, None, None, None


# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def cross_scan_fn(x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
    # y: (B, 4, C, L) | (B, L, 4, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;
    CSF = CrossScanTritonF if WITH_TRITON and x.is_cuda and (not force_torch) else CrossScanF
    with torch.cuda.device(x.device):
        return CSF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)


# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def cross_merge_fn(y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    # y: (B, 4, C, L) | (B, L, 4, C)
    # x: (B, C, H * W) | (B, H * W, C) | (B, 4, C, H * W) | (B, H * W, 4, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;
    CMF = CrossMergeTritonF if WITH_TRITON and y.is_cuda and (not force_torch) else CrossMergeF
    with torch.cuda.device(y.device):
        return CMF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)


# checks =================================================================

class CHECK:
    def check_csm_triton():
        B, C, H, W = 256, 192, 56, 57
        dtype=torch.float16
        dtype=torch.float32
        x = torch.randn((B, C, H, W), dtype=dtype, device=torch.device("cuda")).requires_grad_(True)
        y = torch.randn((B, 4, C, H, W), dtype=dtype, device=torch.device("cuda")).requires_grad_(True)
        x1 = x.clone().detach().requires_grad_(True)
        y1 = y.clone().detach().requires_grad_(True)

        def cross_scan(x: torch.Tensor):
            B, C, H, W = x.shape
            L = H * W
            xs = torch.stack([
                x.view(B, C, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, C, L),
                torch.flip(x.contiguous().view(B, C, L), dims=[-1]),
                torch.flip(torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, C, L), dims=[-1]),
            ], dim=1).view(B, 4, C, L)
            return xs
        
        def cross_merge(out_y: torch.Tensor):
            B, K, D, H, W = out_y.shape
            L = H * W
            out_y = out_y.view(B, K, D, L)
            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
            return y

        def cross_scan_1b1(x: torch.Tensor):
            B, K, C, H, W = x.shape
            L = H * W
            xs = torch.stack([
                x[:, 0].view(B, C, L),
                torch.transpose(x[:, 1], dim0=2, dim1=3).contiguous().view(B, C, L),
                torch.flip(x[:, 2].contiguous().view(B, C, L), dims=[-1]),
                torch.flip(torch.transpose(x[:, 3], dim0=2, dim1=3).contiguous().view(B, C, L), dims=[-1]),
            ], dim=1).view(B, 4, C, L)
            return xs
        
        def unidi_scan(x):
            B, C, H, W = x.shape
            x = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
            return x
        
        def unidi_merge(ys):
            B, K, C, H, W = ys.shape
            return ys.view(B, 4, -1, H * W).sum(1)

        def bidi_scan(x):
            B, C, H, W = x.shape
            x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
            x = torch.cat([x, x.flip(dims=[-1])], dim=1)
            return x
        
        def bidi_merge(ys):
            B, K, D, H, W = ys.shape
            ys = ys.view(B, K, D, -1)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            return ys.contiguous().sum(1)

        if True:
            res0 = triton.testing.do_bench(lambda :cross_scan(x))
            res1 = triton.testing.do_bench(lambda :cross_scan_fn(x, True, True, False))
            # res2 = triton.testing.do_bench(lambda :CrossScanTriton.apply(x))
            res3 = triton.testing.do_bench(lambda :cross_merge(y))
            res4 = triton.testing.do_bench(lambda :cross_merge_fn(y, True, True, False))
            # res5 = triton.testing.do_bench(lambda :CrossMergeTriton.apply(y))
            # print(res0, res1, res2, res3, res4, res5)
            print(res0, res1, res3, res4)
            res0 = triton.testing.do_bench(lambda :cross_scan(x).sum().backward())
            res1 = triton.testing.do_bench(lambda :cross_scan_fn(x, True, True, False).sum().backward())
            # res2 = triton.testing.do_bench(lambda :CrossScanTriton.apply(x).sum().backward())
            res3 = triton.testing.do_bench(lambda :cross_merge(y).sum().backward())
            res4 = triton.testing.do_bench(lambda :cross_merge_fn(y, True, True, False).sum().backward())
            # res5 = triton.testing.do_bench(lambda :CrossMergeTriton.apply(y).sum().backward())
            # print(res0, res1, res2, res3, res4, res5)
            print(res0, res1, res3, res4)

        print("test cross scan")
        for (cs0, cm0, cs1, cm1) in [
            # channel_first -> channel_first
            (cross_scan, cross_merge, cross_scan_fn, cross_merge_fn),
            (unidi_scan, unidi_merge, lambda x: cross_scan_fn(x, scans=1), lambda x: cross_merge_fn(x, scans=1)),
            (bidi_scan, bidi_merge, lambda x: cross_scan_fn(x, scans=2), lambda x: cross_merge_fn(x, scans=2)),
            
            # flex: BLC->BCL; BCL->BLC; BLC->BLC;
            (cross_scan, cross_merge, lambda x: cross_scan_fn(x.permute(0, 2, 3, 1), in_channel_first=False), lambda x: cross_merge_fn(x, in_channel_first=False).permute(0, 2, 1)),
            (cross_scan, cross_merge, lambda x: cross_scan_fn(x, out_channel_first=False).permute(0, 2, 3, 1), lambda x: cross_merge_fn(x.permute(0, 3, 4, 1, 2), out_channel_first=False)),
            (cross_scan, cross_merge, lambda x: cross_scan_fn(x.permute(0, 2, 3, 1), in_channel_first=False, out_channel_first=False).permute(0, 2, 3, 1), lambda x: cross_merge_fn(x.permute(0, 3, 4, 1, 2), in_channel_first=False, out_channel_first=False).permute(0, 2, 1)),
            
            # previous
            # (cross_scan, cross_merge, lambda x: CrossScanTriton.apply(x), lambda x: CrossMergeTriton.apply(x)),
            # (unidi_scan, unidi_merge, lambda x: getCSM(1)[0].apply(x), lambda x: getCSM(1)[1].apply(x)),
            # (bidi_scan, bidi_merge, lambda x: getCSM(2)[0].apply(x), lambda x: getCSM(2)[1].apply(x)),
        ]:
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            o0 = cs0(x)
            o1 = cs1(x1)
            o0.backward(y.view(B, 4, C, H * W))
            o1.backward(y.view(B, 4, C, H * W))
            print((o0 - o1).abs().max())
            print((x.grad - x1.grad).abs().max())
            o0 = cm0(y)
            o1 = cm1(y1)
            o0.backward(x.view(B, C, H * W))
            o1.backward(x.view(B, C, H * W))
            print((o0 - o1).abs().max())
            print((y.grad - y1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            print("===============", flush=True)

        print("test cross scan one by one")
        for (cs0, cs1) in [
            (cross_scan_1b1, lambda x: cross_scan_fn(x, one_by_one=True)),
            # (cross_scan_1b1, lambda x: CrossScanTriton1b1.apply(x)),
        ]:
            o0 = cs0(y)
            o1 = cs1(y1)
            o0.backward(y.view(B, 4, C, H * W))
            o1.backward(y.view(B, 4, C, H * W))
            print((o0 - o1).abs().max())
            print((y.grad - y1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            print("===============", flush=True)

    def check_csm_scan3():
        if False:
            x = torch.arange(0, 16).view(1, 1, 4, 4).cuda()
            out1 = cross_scan_fn(x, scans=3, force_torch=True).view(1, 4, 1, 4, 4)
            out2 = cross_merge_fn(out1, scans=3, force_torch=True).view(1, 1, 4, 4)
            out4 = cross_merge_fn(out1, one_by_one=True, scans=3, force_torch=True).view(1, 4, 1, 4, 4)
            out3 = cross_scan_fn(out4, one_by_one=True, scans=3, force_torch=True).view(1, 4, 1, 4, 4)
            out5 = cross_scan_fn(x.view(1, 4, 4, 1), in_channel_first=False, out_channel_first=False, scans=3, force_torch=True).view(1, 4, 4, 4, 1)
            out6 = cross_merge_fn(out5, in_channel_first=False, out_channel_first=False, scans=3, force_torch=True).view(1, 4, 4, 1)
            out8 = cross_merge_fn(out5, in_channel_first=False, out_channel_first=False, one_by_one=True, scans=3, force_torch=True).view(1, 4, 4, 4, 1)
            out7 = cross_scan_fn(out8, in_channel_first=False, out_channel_first=False, one_by_one=True, scans=3, force_torch=True).view(1, 4, 4, 4, 1)
            print(out1.view(4, -1))
            print(out2.view(-1))
            print(out3.view(4, -1))
            print(out4.view(4, -1))
            print(out5.view(-1, 4).t())
            print(out6.view(-1))
            print(out7.view(-1, 4).t())
            print(out8.view(-1, 4).t())

        B, C, H, W = 27, 253, 57, 58
        x = torch.randn((B, C, H, W)).cuda()

        for scans in [0, 1, 2, 3]:
            o1 = cross_scan_fn(x, scans=scans, force_torch=True).view(B, 4, C, H, W)
            print((cross_scan_fn(x, scans=scans) == cross_scan_fn(x, scans=scans, force_torch=True)).all())
            print((cross_merge_fn(o1, scans=scans) == cross_merge_fn(o1, scans=scans, force_torch=True)).all())

            kwargs = dict(in_channel_first=False, out_channel_first=False)
            x2 = x.permute(0, 2, 3, 1).contiguous()
            o2 = o1.permute(0, 3, 4, 1, 2).contiguous()
            print((cross_scan_fn(x, scans=scans, **kwargs) == cross_scan_fn(x, scans=scans, force_torch=True, **kwargs)).all())
            print((cross_merge_fn(o2, scans=scans, **kwargs) == cross_merge_fn(o2, scans=scans, force_torch=True, **kwargs)).all())            

        breakpoint()


# =====================================================================
# Triton Selective Scan Implementation
# =====================================================================

def selective_scan_ref_csm(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, chunk_size=1024):
    """
    Pure-PyTorch fallback implementation of selective scan.
    """
    import torch.nn.functional as F
    B_batch, D_dim, L = u.shape
    N = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)

    G = B.shape[1] if B.dim() == 4 else 1
    d_per_g = D_dim // G
    A_exp = A.unsqueeze(0).unsqueeze(2)

    num_chunks = (L + chunk_size - 1) // chunk_size
    ys = []
    x_prev = torch.zeros(B_batch, D_dim, N, device=u.device, dtype=u.dtype)

    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, L)
        Lc = end - start

        u_c = u[:, :, start:end]
        delta_c = delta[:, :, start:end]

        if B.dim() == 4:
            b_c = B[:, :, :, start:end].unsqueeze(2).expand(B_batch, G, d_per_g, N, Lc).reshape(B_batch, D_dim, N, Lc).permute(0, 1, 3, 2)
            c_c = C[:, :, :, start:end].unsqueeze(2).expand(B_batch, G, d_per_g, N, Lc).reshape(B_batch, D_dim, N, Lc).permute(0, 1, 3, 2)
        else:
            b_c = B[:, :, start:end].unsqueeze(2).expand(B_batch, D_dim, 1, Lc).permute(0, 1, 3, 2)
            c_c = C[:, :, start:end].unsqueeze(2).expand(B_batch, D_dim, 1, Lc).permute(0, 1, 3, 2)

        delta_A_c = delta_c.unsqueeze(-1) * A_exp
        log_a_cum = torch.cumsum(delta_A_c, dim=2)
        log_a_cum = torch.clamp(log_a_cum, min=-40.0, max=0.0)
        a_cum = torch.exp(log_a_cum)

        du_c = (delta_c * u_c).unsqueeze(-1)
        v_scaled = (b_c * torch.exp(-log_a_cum)) * du_c
        x_intra = a_cum * (torch.cumsum(v_scaled, dim=2) + x_prev.unsqueeze(2))

        y_c = (x_intra * c_c).sum(dim=-1)
        ys.append(y_c)
        x_prev = x_intra[:, :, -1]

    y = torch.cat(ys, dim=-1)
    if D is not None and isinstance(D, torch.Tensor):
        y = y + u * D.unsqueeze(0).unsqueeze(-1)

    return y


if WITH_TRITON:
    @triton.jit
    def _selective_scan_fwd_kernel(
        u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, D_ptr, delta_bias_ptr,
        out_ptr, h_out_ptr,
        stride_u_b, stride_u_d, stride_u_l,
        stride_delta_b, stride_delta_d, stride_delta_l,
        stride_A_d, stride_A_n,
        stride_B_b, stride_B_g, stride_B_n, stride_B_l,
        stride_C_b, stride_C_g, stride_C_n, stride_C_l,
        stride_D_d,
        stride_bias_d,
        stride_out_b, stride_out_d, stride_out_l,
        stride_h_b, stride_h_d, stride_h_l, stride_h_n,
        L: tl.constexpr,
        D_val: tl.constexpr,
        G_val: tl.constexpr,
        d_per_g: tl.constexpr,
        HAS_D: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        DELTA_SOFTPLUS: tl.constexpr,
        SAVE_H: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // D_val
        d = pid % D_val
        g = d // d_per_g

        offs_n = tl.arange(0, 16)
        A_val = tl.load(A_ptr + d * stride_A_d + offs_n * stride_A_n).to(tl.float32)

        bias_val = 0.0
        if HAS_BIAS:
            bias_val = tl.load(delta_bias_ptr + d * stride_bias_d).to(tl.float32)

        d_skip = 0.0
        if HAS_D:
            d_skip = tl.load(D_ptr + d * stride_D_d).to(tl.float32)

        h = tl.zeros((16,), dtype=tl.float32)

        u_base = u_ptr + b * stride_u_b + d * stride_u_d
        delta_base = delta_ptr + b * stride_delta_b + d * stride_delta_d
        B_base = B_ptr + b * stride_B_b + g * stride_B_g + offs_n * stride_B_n
        C_base = C_ptr + b * stride_C_b + g * stride_C_g + offs_n * stride_C_n
        out_base = out_ptr + b * stride_out_b + d * stride_out_d
        h_base = h_out_ptr + b * stride_h_b + d * stride_h_d + offs_n * stride_h_n

        for l in range(0, L):
            u_val = tl.load(u_base + l * stride_u_l).to(tl.float32)
            dt = tl.load(delta_base + l * stride_delta_l).to(tl.float32)

            if HAS_BIAS:
                dt = dt + bias_val

            if DELTA_SOFTPLUS:
                dt = tl.where(dt > 20.0, dt, tl.log(1.0 + tl.exp(dt)))

            b_vec = tl.load(B_base + l * stride_B_l).to(tl.float32)
            c_vec = tl.load(C_base + l * stride_C_l).to(tl.float32)

            a_vec = tl.exp(dt * A_val)
            h = a_vec * h + (dt * u_val) * b_vec
            y_val = tl.sum(h * c_vec)

            if HAS_D:
                y_val = y_val + u_val * d_skip

            tl.store(out_base + l * stride_out_l, y_val)
            if SAVE_H:
                tl.store(h_base + l * stride_h_l, h)


    @triton.jit
    def _selective_scan_bwd_kernel(
        dy_ptr, u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, D_ptr, delta_bias_ptr, h_ptr,
        du_ptr, ddelta_ptr, dA_ptr, dB_ptr, dC_ptr, dD_ptr, dbias_ptr,
        stride_dy_b, stride_dy_d, stride_dy_l,
        stride_u_b, stride_u_d, stride_u_l,
        stride_delta_b, stride_delta_d, stride_delta_l,
        stride_A_d, stride_A_n,
        stride_B_b, stride_B_g, stride_B_n, stride_B_l,
        stride_C_b, stride_C_g, stride_C_n, stride_C_l,
        stride_D_d,
        stride_bias_d,
        stride_h_b, stride_h_d, stride_h_l, stride_h_n,
        stride_du_b, stride_du_d, stride_du_l,
        stride_ddelta_b, stride_ddelta_d, stride_ddelta_l,
        stride_dA_d, stride_dA_n,
        stride_dB_b, stride_dB_g, stride_dB_n, stride_dB_l,
        stride_dC_b, stride_dC_g, stride_dC_n, stride_dC_l,
        stride_dD_d,
        stride_dbias_d,
        L: tl.constexpr,
        D_val: tl.constexpr,
        G_val: tl.constexpr,
        d_per_g: tl.constexpr,
        HAS_D: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        DELTA_SOFTPLUS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // D_val
        d = pid % D_val
        g = d // d_per_g

        offs_n = tl.arange(0, 16)
        A_val = tl.load(A_ptr + d * stride_A_d + offs_n * stride_A_n).to(tl.float32)

        bias_val = 0.0
        if HAS_BIAS:
            bias_val = tl.load(delta_bias_ptr + d * stride_bias_d).to(tl.float32)

        d_skip = 0.0
        if HAS_D:
            d_skip = tl.load(D_ptr + d * stride_D_d).to(tl.float32)

        dh = tl.zeros((16,), dtype=tl.float32)
        dA_acc = tl.zeros((16,), dtype=tl.float32)
        dbias_acc = 0.0
        dD_acc = 0.0

        dy_base = dy_ptr + b * stride_dy_b + d * stride_dy_d
        u_base = u_ptr + b * stride_u_b + d * stride_u_d
        delta_base = delta_ptr + b * stride_delta_b + d * stride_delta_d
        B_base = B_ptr + b * stride_B_b + g * stride_B_g + offs_n * stride_B_n
        C_base = C_ptr + b * stride_C_b + g * stride_C_g + offs_n * stride_C_n
        h_base = h_ptr + b * stride_h_b + d * stride_h_d + offs_n * stride_h_n

        du_base = du_ptr + b * stride_du_b + d * stride_du_d
        ddelta_base = ddelta_ptr + b * stride_ddelta_b + d * stride_ddelta_d
        dB_base = dB_ptr + b * stride_dB_b + g * stride_dB_g + offs_n * stride_dB_n
        dC_base = dC_ptr + b * stride_dC_b + g * stride_dC_g + offs_n * stride_dC_n

        for l_rev in range(0, L):
            l = L - 1 - l_rev
            dy_val = tl.load(dy_base + l * stride_dy_l).to(tl.float32)
            u_val = tl.load(u_base + l * stride_u_l).to(tl.float32)
            dt_orig = tl.load(delta_base + l * stride_delta_l).to(tl.float32)
            dt_val = dt_orig + bias_val if HAS_BIAS else dt_orig

            if DELTA_SOFTPLUS:
                dt = tl.where(dt_val > 20.0, dt_val, tl.log(1.0 + tl.exp(dt_val)))
            else:
                dt = dt_val

            b_vec = tl.load(B_base + l * stride_B_l).to(tl.float32)
            c_vec = tl.load(C_base + l * stride_C_l).to(tl.float32)
            h_curr = tl.load(h_base + l * stride_h_l).to(tl.float32)
            h_prev = tl.zeros((16,), dtype=tl.float32)
            if l > 0:
                h_prev = tl.load(h_base + (l - 1) * stride_h_l).to(tl.float32)

            # 1. Update dh
            dh = dh + dy_val * c_vec

            # 2. dC = dy * h_curr (atomic add across channels sharing g)
            dC_val = dy_val * h_curr
            tl.atomic_add(dC_base + l * stride_dC_l, dC_val)

            # 3. du = dt * (dh * B) + D * dy
            du_val = dt * tl.sum(dh * b_vec)
            if HAS_D:
                du_val = du_val + dy_val * d_skip
                dD_acc = dD_acc + dy_val * u_val
            tl.store(du_base + l * stride_du_l, du_val)

            # 4. dB = dt * u * dh (atomic add across channels sharing g)
            dB_val = (dt * u_val) * dh
            tl.atomic_add(dB_base + l * stride_dB_l, dB_val)

            # 5. dA and ddelta
            a_vec = tl.exp(dt * A_val)
            d_exp = a_vec * dh * h_prev
            dA_acc = dA_acc + dt * d_exp

            d_dt = tl.sum(A_val * d_exp + (u_val * b_vec) * dh)
            if DELTA_SOFTPLUS:
                sig = tl.sigmoid(dt_val)
                d_dt_raw = d_dt * sig
            else:
                d_dt_raw = d_dt

            tl.store(ddelta_base + l * stride_ddelta_l, d_dt_raw)
            if HAS_BIAS:
                dbias_acc = dbias_acc + d_dt_raw

            # 6. dh for previous step
            dh = a_vec * dh

        # Atomic add across batch elements
        dA_base = dA_ptr + d * stride_dA_d + offs_n * stride_dA_n
        tl.atomic_add(dA_base, dA_acc)

        if HAS_BIAS:
            tl.atomic_add(dbias_ptr + d * stride_dbias_d, dbias_acc)

        if HAS_D:
            tl.atomic_add(dD_ptr + d * stride_dD_d, dD_acc)


    class _TritonSelectiveScanAutograd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False):
            B_batch, D_dim, L = u.shape
            N = A.shape[1]
            assert N == 16, f"Only N=16 supported, got {N}"
            G = B.shape[1] if B.dim() == 4 else 1
            d_per_g = D_dim // G

            out = torch.empty_like(u)
            h_saved = torch.empty(B_batch, D_dim, L, N, device=u.device, dtype=torch.float32)

            grid = (B_batch * D_dim,)
            _selective_scan_fwd_kernel[grid](
                u, delta, A, B, C,
                D if D is not None else u,
                delta_bias if delta_bias is not None else u,
                out, h_saved,
                u.stride(0), u.stride(1), u.stride(2),
                delta.stride(0), delta.stride(1), delta.stride(2),
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1) if B.dim() == 4 else 0, B.stride(2) if B.dim() == 4 else B.stride(1), B.stride(-1),
                C.stride(0), C.stride(1) if C.dim() == 4 else 0, C.stride(2) if C.dim() == 4 else C.stride(1), C.stride(-1),
                D.stride(0) if D is not None else 0,
                delta_bias.stride(0) if delta_bias is not None else 0,
                out.stride(0), out.stride(1), out.stride(2),
                h_saved.stride(0), h_saved.stride(1), h_saved.stride(2), h_saved.stride(3),
                L=L,
                D_val=D_dim,
                G_val=G,
                d_per_g=d_per_g,
                HAS_D=(D is not None),
                HAS_BIAS=(delta_bias is not None),
                DELTA_SOFTPLUS=delta_softplus,
                SAVE_H=True,
            )

            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, h_saved)
            ctx.delta_softplus = delta_softplus
            ctx.G = G
            ctx.d_per_g = d_per_g
            return out

        @staticmethod
        def backward(ctx, dy):
            u, delta, A, B, C, D, delta_bias, h_saved = ctx.saved_tensors
            B_batch, D_dim, L = u.shape
            N = A.shape[1]
            G = ctx.G
            d_per_g = ctx.d_per_g

            du = torch.empty_like(u)
            ddelta = torch.empty_like(delta)
            dA = torch.zeros_like(A, dtype=torch.float32)
            dB = torch.zeros_like(B, dtype=torch.float32)
            dC = torch.zeros_like(C, dtype=torch.float32)
            dD = torch.zeros_like(D, dtype=torch.float32) if D is not None else None
            ddelta_bias = torch.zeros_like(delta_bias, dtype=torch.float32) if delta_bias is not None else None

            grid = (B_batch * D_dim,)
            _selective_scan_bwd_kernel[grid](
                dy, u, delta, A, B, C,
                D if D is not None else u,
                delta_bias if delta_bias is not None else u,
                h_saved,
                du, ddelta, dA, dB, dC,
                dD if dD is not None else u,
                ddelta_bias if ddelta_bias is not None else u,
                dy.stride(0), dy.stride(1), dy.stride(2),
                u.stride(0), u.stride(1), u.stride(2),
                delta.stride(0), delta.stride(1), delta.stride(2),
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1) if B.dim() == 4 else 0, B.stride(2) if B.dim() == 4 else B.stride(1), B.stride(-1),
                C.stride(0), C.stride(1) if C.dim() == 4 else 0, C.stride(2) if C.dim() == 4 else C.stride(1), C.stride(-1),
                D.stride(0) if D is not None else 0,
                delta_bias.stride(0) if delta_bias is not None else 0,
                h_saved.stride(0), h_saved.stride(1), h_saved.stride(2), h_saved.stride(3),
                du.stride(0), du.stride(1), du.stride(2),
                ddelta.stride(0), ddelta.stride(1), ddelta.stride(2),
                dA.stride(0), dA.stride(1),
                dB.stride(0), dB.stride(1) if dB.dim() == 4 else 0, dB.stride(2) if dB.dim() == 4 else dB.stride(1), dB.stride(-1),
                dC.stride(0), dC.stride(1) if dC.dim() == 4 else 0, dC.stride(2) if dC.dim() == 4 else dC.stride(1), dC.stride(-1),
                dD.stride(0) if dD is not None else 0,
                ddelta_bias.stride(0) if ddelta_bias is not None else 0,
                L=L,
                D_val=D_dim,
                G_val=G,
                d_per_g=d_per_g,
                HAS_D=(D is not None),
                HAS_BIAS=(delta_bias is not None),
                DELTA_SOFTPLUS=ctx.delta_softplus,
            )

            return (
                du,
                ddelta,
                dA.to(A.dtype),
                dB.to(B.dtype),
                dC.to(C.dtype),
                dD.to(D.dtype) if dD is not None else None,
                ddelta_bias.to(delta_bias.dtype) if ddelta_bias is not None else None,
                None,
            )


def selective_scan_fn(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False):
    """
    Triton-accelerated selective scan function with automatic fallback to pure PyTorch.
    Compatible with Triton >= 3.0.0.
    """
    if WITH_TRITON and u.is_cuda and A.shape[1] == 16:
        return _TritonSelectiveScanAutograd.apply(u, delta, A, B, C, D, delta_bias, delta_softplus)
    else:
        return selective_scan_ref_csm(u, delta, A, B, C, D, delta_bias, delta_softplus)

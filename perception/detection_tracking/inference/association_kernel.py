"""Fused rotated overlap for eligible pairs; imported only by the CUDA backend.

One program clips two rectangles, then computes IoU/DIoU/GIoU. Float64 local
geometry preserves narrow gaps, collinear edges and large-world translations.
Triton JIT caches the kernel; it does not build the detector's CUDA extensions.
"""
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _corners(x, y, length, width, yaw, index):
    sx = tl.where((index == 0) | (index == 3), -0.5, 0.5)
    sy = tl.where(index < 2, -0.5, 0.5)
    c, s = libdevice.cos(yaw), libdevice.sin(yaw)
    return x+sx*length*c-sy*width*s, y+sx*length*s+sy*width*c


@tr.jit
def overlap_kernel(Boxes, Pairs, Result, BEV: tl.constexpr, METRIC: tl.constexpr):
    pair = tl.program_id(0)
    ia, ib = tl.load(Pairs+pair*2), tl.load(Pairs+pair*2+1)
    a, b = Boxes+ia*7, Boxes+ib*7
    # Subtract in float64 before any geometry. Each pair has its own local origin.
    dx, dy, dz = tl.load(b)-tl.load(a), tl.load(b+1)-tl.load(a+1), tl.load(b+2)-tl.load(a+2)
    al, aw, ah, ay = tl.load(a+3), tl.load(a+4), tl.load(a+5), tl.load(a+6)
    bl, bw, bh, by = tl.load(b+3), tl.load(b+4), tl.load(b+5), tl.load(b+6)
    index = tl.arange(0, 8)
    ax, ayc = _corners(0., 0., al, aw, ay, index % 4)
    bx, byc = _corners(dx, dy, bl, bw, by, index % 4)
    px, py = ax, ayc
    count = 4
    # Sutherland-Hodgman clipping: two rectangles have at most eight vertices.
    candidate = tl.arange(0, 16)
    current = candidate // 2
    for edge in range(4):
        ex, ey = _corners(dx, dy, bl, bw, by, edge)
        nx, ny = _corners(dx, dy, bl, bw, by, (edge+1) % 4)
        cx, cy = tl.gather(px, current, 0), tl.gather(py, current, 0)
        previous = (current+tl.maximum(count, 1)-1) % tl.maximum(count, 1)
        vx, vy = tl.gather(px, previous, 0), tl.gather(py, previous, 0)
        before = (nx-ex)*(vy-ey)-(ny-ey)*(vx-ex)
        after = (nx-ex)*(cy-ey)-(ny-ey)*(cx-ex)
        crossing = ((before >= -1.e-9) != (after >= -1.e-9)) & (tl.abs(before-after) > 1.e-12)
        fraction = before/tl.where(crossing, before-after, 1.)
        ix, iy = vx+fraction*(cx-vx), vy+fraction*(cy-vy)
        valid = (current < count) & tl.where(candidate % 2 == 0, crossing, after >= -1.e-9)
        xx, yy = tl.where(candidate % 2 == 0, ix, cx), tl.where(candidate % 2 == 0, iy, cy)
        rank = tl.cumsum(valid.to(tl.int32))-1
        select = valid[:, None] & (rank[:, None] == index[None, :])
        px = tl.sum(tl.where(select, xx[:, None], 0.), 0)
        py = tl.sum(tl.where(select, yy[:, None], 0.), 0)
        count = tl.sum(valid.to(tl.int32), 0)
    following = (index+1) % tl.maximum(count, 1)
    cross = px*tl.gather(py, following, 0)-py*tl.gather(px, following, 0)
    intersection = tl.abs(tl.sum(tl.where(index < count, cross, 0.), 0))/2
    va, vb = al*aw, bl*bw
    height = tl.maximum(ah/2, dz+bh/2)-tl.minimum(-ah/2, dz-bh/2)
    if not BEV:
        overlap_height = tl.maximum(0., tl.minimum(ah/2, dz+bh/2)-tl.maximum(-ah/2, dz-bh/2))
        intersection *= overlap_height
        va, vb = va*ah, vb*bh
    union = va+vb-intersection
    score = intersection/tl.maximum(union, 1.e-12)
    if METRIC == 'diou':
        lowx, lowy = tl.minimum(tl.min(ax, 0), tl.min(bx, 0)), tl.minimum(tl.min(ayc, 0), tl.min(byc, 0))
        highx, highy = tl.maximum(tl.max(ax, 0), tl.max(bx, 0)), tl.maximum(tl.max(ayc, 0), tl.max(byc, 0))
        diagonal = (highx-lowx)*(highx-lowx)+(highy-lowy)*(highy-lowy)
        distance = dx*dx+dy*dy
        if not BEV:
            diagonal += height*height
            distance += dz*dz
        score -= distance/tl.maximum(diagonal, 1.e-12)
    elif METRIC == 'giou':
        # An oriented hull edge has all eight corners on its left. Exclude
        # duplicate coordinates and edges that skip a collinear corner.
        hx, hy = tl.where(index < 4, ax, bx), tl.where(index < 4, ayc, byc)
        vx, vy = hx[None, :]-hx[:, None], hy[None, :]-hy[:, None]
        length = vx*vx+vy*vy
        unique = ~tl.sum(((length <= 1.e-16) & (index[None, :] < index[:, None])).to(tl.int32), 1).to(tl.int1)
        cross = vx[:, :, None]*vy[:, None, :]-vy[:, :, None]*vx[:, None, :]
        dot = vx[:, :, None]*vx[:, None, :]+vy[:, :, None]*vy[:, None, :]
        between = tl.sum(((tl.abs(cross) <= 1.e-10) & (dot > 1.e-10)
                          & (dot < length[:, :, None]-1.e-10)).to(tl.int32), 2) > 0
        boundary = (tl.min(cross, 2) >= -1.e-10) & ~between & (length > 1.e-16) & unique[:, None] & unique[None, :]
        wedge = hx[:, None]*hy[None, :]-hy[:, None]*hx[None, :]
        enclosure = tl.abs(tl.sum(tl.sum(tl.where(boundary, wedge, 0.), 1), 0))/2
        if not BEV:
            enclosure *= height
        score -= (enclosure-union)/tl.maximum(enclosure, 1.e-12)
    tl.store(Result+pair, score)

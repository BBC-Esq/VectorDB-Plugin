import numpy as np

_X_OVERLAP = 0.2
_Y_OVERLAP = 0.5
_MIN_ANGLE = 1.0
_MAX_ANGLE = 45.0
_SPANNING_FRAC = 0.5
_MARGIN_FRAC = 0.125
_MIN_BODY_BOXES = 8
_MIN_GROUP_BOXES = 4
_GUTTER_FRAC = 0.015
_GROUP_SPAN_FRAC = 0.22
_GROUP_FILL_FRAC = 0.55
_GROUP_HEIGHT_FRAC = 0.4
_MAX_GROUPS = 3


def _to_boxes(polys):
    if len(polys) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    pts = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polys]
    return np.asarray([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in pts])


def _overlap_ratios(starts, ends):
    inter = np.minimum(ends[:, None], ends[None, :]) - np.maximum(starts[:, None], starts[None, :])
    min_len = np.minimum(ends - starts, (ends - starts)[:, None])
    return np.clip(inter, 0, None) / np.clip(min_len, 1e-9, None)


def _strict_rank(primary, secondary):
    order = np.lexsort((np.arange(primary.shape[0]), secondary, primary))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.shape[0])
    return ranks


def _page_angle(polys):
    pts = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polys]
    if len(pts) == 0 or any(p.shape[0] != 4 for p in pts):
        return 0.0
    a = np.stack(pts)
    xleft = a[:, 0, 0] + a[:, 3, 0]
    yleft = a[:, 0, 1] + a[:, 3, 1]
    xright = a[:, 1, 0] + a[:, 2, 0]
    yright = a[:, 1, 1] + a[:, 2, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        angles = np.arctan((yleft - yright) / (xright - xleft)) * 180 / np.pi
    angles = angles[np.isfinite(angles)]
    return float(np.median(angles)) if angles.size > 0 else 0.0


def _deskew(polys, angle):
    pts = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polys]
    theta = np.deg2rad(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    center = np.concatenate(pts, axis=0).mean(axis=0)
    return [(p - center) @ rot.T + center for p in pts]


def _structure(boxes):
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    span_lo, span_hi = float(x0.min()), float(x1.max())
    span = span_hi - span_lo
    page_y0, page_y1 = float(y0.min()), float(y1.max())
    page_h = page_y1 - page_y0
    if span <= 0 or page_h <= 0:
        return 'ambiguous'
    w = x1 - x0
    yc = (y0 + y1) / 2
    body = (w <= _SPANNING_FRAC * span) & (yc >= page_y0 + _MARGIN_FRAC * page_h) & (yc <= page_y1 - _MARGIN_FRAC * page_h)
    idx = np.flatnonzero(body)
    if idx.size < _MIN_BODY_BOXES:
        return 'single'
    order = idx[np.argsort(x0[idx])]
    reach = float(x1[order[0]])
    cuts = []
    for i in order[1:]:
        if float(x0[i]) - reach >= _GUTTER_FRAC * span:
            cuts.append((reach + float(x0[i])) / 2)
        reach = max(reach, float(x1[i]))
    if not cuts:
        return 'single'
    if len(cuts) > _MAX_GROUPS - 1:
        return 'ambiguous'
    bounds = [span_lo] + cuts + [span_hi + 1.0]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        g = idx[(x0[idx] >= lo) & (x0[idx] < hi)]
        if g.size < _MIN_GROUP_BOXES:
            return 'ambiguous'
        gspan = float(x1[g].max() - x0[g].min())
        if gspan < _GROUP_SPAN_FRAC * span:
            return 'ambiguous'
        if float(np.median(w[g])) < _GROUP_FILL_FRAC * max(gspan, 1e-9):
            return 'ambiguous'
        if float(y1[g].max() - y0[g].min()) < _GROUP_HEIGHT_FRAC * page_h:
            return 'ambiguous'
    return 'columns'


def _topological_order(boxes):
    num_boxes = boxes.shape[0]
    if num_boxes <= 1:
        return list(range(num_boxes))
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x_overlap = _overlap_ratios(x0, x1)
    y_overlap = _overlap_ratios(y0, y1)
    x_rank = _strict_rank(x0, x1)
    y_rank = _strict_rank(y0, y1)
    is_above = y_rank[:, None] < y_rank[None, :]
    is_left = x_rank[:, None] < x_rank[None, :]
    edges = ((x_overlap > _X_OVERLAP) & is_above) | (
        (x_overlap <= _X_OVERLAP) & (y_overlap > _Y_OVERLAP) & is_left
    )
    np.fill_diagonal(edges, False)
    in_degree = edges.sum(axis=0)
    emitted = np.zeros(num_boxes, dtype=bool)
    order = []
    last = -1
    page_width = float(x1.max() - x0.min()) or 1.0
    spanning = (x1 - x0) > _SPANNING_FRAC * page_width
    parent = np.arange(num_boxes)

    def _find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    col_edges = (x_overlap > _X_OVERLAP) & ~spanning[:, None] & ~spanning[None, :]
    for i, j in np.argwhere(np.triu(col_edges, 1)):
        ri, rj = _find(int(i)), _find(int(j))
        if ri != rj:
            parent[ri] = rj
    component = np.array([_find(i) for i in range(num_boxes)])

    while len(order) < num_boxes:
        ready = np.flatnonzero((in_degree == 0) & ~emitted)
        if ready.size == 0:
            ready = np.flatnonzero(~emitted)
            candidates = ready
        else:
            candidates = (
                ready[(x_overlap[last, ready] > _X_OVERLAP) & (y0[ready] >= y0[last])]
                if last >= 0
                else np.empty(0, dtype=int)
            )
            if candidates.size == 0 and last >= 0:
                candidates = ready[y_overlap[last, ready] > _Y_OVERLAP]
            if candidates.size == 0 and last >= 0 and not spanning[last]:
                same_column = ready[component[ready] == component[last]]
                candidates = same_column if same_column.size else ready
            elif candidates.size == 0:
                candidates = ready
        next_idx = int(candidates[np.lexsort((x0[candidates], y0[candidates]))[0]])
        order.append(next_idx)
        emitted[next_idx] = True
        in_degree = in_degree - edges[next_idx]
        last = next_idx

    return order


def column_reading_order(polys):
    boxes = _to_boxes(polys)
    if boxes.shape[0] < 2:
        return None
    angle = _page_angle(polys)
    if _MIN_ANGLE <= abs(angle) < _MAX_ANGLE:
        boxes = _to_boxes(_deskew(polys, angle))
    if _structure(boxes) == 'ambiguous':
        return None
    return _topological_order(boxes)

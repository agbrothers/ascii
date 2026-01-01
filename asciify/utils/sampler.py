import numpy as np


def _neighbors4(r, c, H, W):
    if r > 0:     yield r - 1, c
    if r + 1 < H: yield r + 1, c
    if c > 0:     yield r, c - 1
    if c + 1 < W: yield r, c + 1


class MoldFrontierSampler:

    def __init__(self, shape, seed_rc, *, fraction=1/3, rng=None):
        self.H, self.W = shape
        self.rng = np.random.default_rng() if rng is None else rng
        self.sampled = np.zeros((self.H, self.W), dtype=bool)
        self.fraction = float(fraction)

        ## STORE FRONTIER AS A 1D IDX SET FOR SPEED
        ## (r, c) <-> idx = r*W + c
        self.frontier = set()
        r0, c0 = map(int, seed_rc)
        self.add_sampled([(r0, c0)])
        return

    def rc_to_idx(self, r, c):
        return r * self.W + c

    def add_to_frontier_if_valid(self, r, c):
        if not self.sampled[r, c]:
            self.frontier.add(self.rc_to_idx(r, c))
        return

    def add_sampled(self, points):
        ## UPDATE THE FRONTIER
        for r, c in points:
            if self.sampled[r, c]:
                continue
            self.sampled[r, c] = True

            ## REMOVE SAMPLED PTS FROM THE FRONTIER
            self.frontier.discard(self.rc_to_idx(r, c))

            ## ADD UNSAMPLED NEIGHBORS TO THE FRONTIER
            for rr, cc in _neighbors4(r, c, self.H, self.W):
                self.add_to_frontier_if_valid(rr, cc)
        return

    def step(self, *, min_new=1, max_new=None):
        ## SAMPLE `self.fraction` OF THE FRONTIER AS NEW POINTS
        fsize = len(self.frontier)
        if fsize == 0:
            return np.empty((0, 2), dtype=np.int32)

        m = int(np.floor(fsize * self.fraction))
        m = max(int(min_new), m)

        if max_new is not None:
            m = min(m, int(max_new))
        m = min(m, fsize)

        ## SAMPLE `m` UNIQUE IDXS FROM THE FRONTIER
        frontier_list = np.fromiter(self.frontier, dtype=np.int64, count=fsize)
        chosen = self.rng.choice(frontier_list, size=m, replace=False)

        rr = (chosen // self.W).astype(np.int32)
        cc = (chosen % self.W).astype(np.int32)

        self.add_sampled(zip(rr.tolist(), cc.tolist()))
        return np.stack([rr, cc], axis=1)

    def done(self):
        return self.sampled.all() or len(self.frontier) == 0

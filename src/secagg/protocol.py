"""
Post-quantum sparse secure aggregation with sketch filtering, simulated as one
server + N clients in one process. Follows Algorithms 1-2 / Sec. III-IV of the
paper:

  Round t
   0. Key agreement on a connected k-regular graph, k = max(2, ceil(log2 N)):
      every client draws an ephemeral ML-KEM-768 key, the encapsulation key and
      every ciphertext are ML-DSA-65 signed and VERIFIED by the receiver.
      Pairwise seed s_ij = SHA3-256(ss || ct || i || j || t).
   1. Client i clips its update (g_bar = clip(g, C)), encodes it in Z_q, q = 2^32,
      with phi fractional bits, and uploads
         y_i = enc(g_bar_i) + PRG(b_i) + sum_{j>i} PRG(s_ij) - sum_{j<i} PRG(s_ij)  (mod q)
      PRG = SHAKE-256 (uniform masks -> one-time-pad hiding). The self-mask seed
      b_i is l-of-k Shamir shared to its neighbours (shares encrypted under s_ij).
      The upload is signed together with H(y_i) (commitment).
   2. Commit-then-project: only after all uploads are in, the server draws fresh
      r_S and fixes P = PRG(H(t || H(uploads) || r_S)).
   3. Clients send signed sketches s_i = clip(P g_bar_i, C_s) + N(0, sigma^2 I).
   4. Server filter -> accepted set A. Rejected or dropped clients are treated
      as dropped.
   5. Unmasking: for i in A the online neighbours reveal their Shamir shares of
      b_i (>= l needed; checked against the committed H(b_i)); for j not in A the
      accepted neighbours reveal s_ij (checked against the committed H(s_ij)).
      Never both for the same client.
   6. Consistency check  ||P U - sum_A s_i|| <= |A| sigma (sqrt(m) + sqrt(2 ln(N/gamma)));
      the round is aborted (no model update) on failure.

All signatures, KEM operations, PRG expansions, Shamir operations and the check
are executed for real; only the network is replaced by function calls.
r_S and b_i are drawn from the run's seeded RNG so that runs are reproducible
(the simulated clients do not adapt to P, so this does not help any attack).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
import oqs

KEM_ALG = "ML-KEM-768"
SIG_ALG = "ML-DSA-65"
MOD_BITS = 32
SHAMIR_P = (1 << 521) - 1  # Mersenne prime > 2^256


# ----------------------------------------------------------------------------
# Graph
# ----------------------------------------------------------------------------
def random_regular_graph(n: int, k: int, seed: int, max_tries: int = 500) -> dict[int, list[int]]:
    """Connected simple k-regular graph: pairing model with retries, falling back to a
    randomly relabelled Harary (circulant) graph, which is always k-connected.
    k is increased by one if n*k is odd."""
    if k >= n - 1:
        return {i: [j for j in range(n) if j != i] for i in range(n)}
    if (n * k) % 2 == 1:
        k += 1
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        stubs = np.repeat(np.arange(n), k)
        rng.shuffle(stubs)
        pairs = stubs.reshape(-1, 2)
        if np.any(pairs[:, 0] == pairs[:, 1]):
            continue
        edges = {tuple(sorted(map(int, p))) for p in pairs}
        if len(edges) != len(pairs):
            continue
        adj: dict[int, set[int]] = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        seen, stack = {0}, [0]
        while stack:
            u = stack.pop()
            for v in adj[u] - seen:
                seen.add(v)
                stack.append(v)
        if len(seen) == n:
            return {i: sorted(adj[i]) for i in range(n)}
    perm = rng.permutation(n)
    adj = {i: set() for i in range(n)}
    offs = list(range(1, k // 2 + 1)) + ([n // 2] if k % 2 else [])
    for i in range(n):
        for o in offs:
            u, v = int(perm[i]), int(perm[(i + o) % n])
            adj[u].add(v)
            adj[v].add(u)
    return {i: sorted(adj[i]) for i in range(n)}


def log_degree(n: int) -> int:
    return max(2, int(np.ceil(np.log2(max(4, n)))))


# ----------------------------------------------------------------------------
# Fixed point in Z_{2^32}, PRG, Shamir
# ----------------------------------------------------------------------------
@dataclass
class FixedPoint:
    frac_bits: int = 16
    clip: float = 5.0  # per-coordinate bound; |g_bar_j| <= ||g_bar|| <= C

    def check_capacity(self, n: int) -> None:
        if n * self.clip * (1 << self.frac_bits) >= (1 << (MOD_BITS - 1)):
            raise ValueError("fixed-point range overflow: lower clip or frac_bits")

    def encode(self, x: np.ndarray) -> np.ndarray:
        xc = np.clip(x.astype(np.float64), -self.clip, self.clip)
        q = np.rint(xc * (1 << self.frac_bits)).astype(np.int64)
        return (q & 0xFFFFFFFF).astype(np.uint32)

    def decode(self, s: np.ndarray) -> np.ndarray:
        v = s.astype(np.int64)
        v[v >= (1 << (MOD_BITS - 1))] -= 1 << MOD_BITS
        return v.astype(np.float64) / (1 << self.frac_bits)


def prg_u32(seed: bytes, label: bytes, d: int) -> np.ndarray:
    return np.frombuffer(hashlib.shake_256(label + seed).digest(4 * d), dtype=np.uint32)


def _msg(*parts) -> bytes:
    return b"|".join(p.to_bytes(8, "big") if isinstance(p, int) else p for p in parts)


def shamir_split(secret: bytes, n_shares: int, threshold: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    s = int.from_bytes(secret, "big")
    coeffs = [s] + [int.from_bytes(rng.bytes(64), "big") % SHAMIR_P for _ in range(threshold - 1)]
    out = []
    for x in range(1, n_shares + 1):
        y = 0
        for c in reversed(coeffs):
            y = (y * x + c) % SHAMIR_P
        out.append((x, y))
    return out


def shamir_combine(shares: list[tuple[int, int]], nbytes: int = 32) -> bytes:
    s = 0
    for i, (xi, yi) in enumerate(shares):
        num, den = 1, 1
        for j, (xj, _) in enumerate(shares):
            if i != j:
                num = num * (-xj) % SHAMIR_P
                den = den * (xi - xj) % SHAMIR_P
        s = (s + yi * num * pow(den, -1, SHAMIR_P)) % SHAMIR_P
    return s.to_bytes(nbytes, "big")


def _xor(data: bytes, key: bytes, label: bytes) -> bytes:
    ks = hashlib.shake_256(label + key).digest(len(data))
    return bytes(a ^ b for a, b in zip(data, ks))


# ----------------------------------------------------------------------------
@dataclass
class RoundStats:
    client_kem_ms: float = 0.0
    client_sign_ms: float = 0.0
    client_verify_ms: float = 0.0
    client_mask_ms: float = 0.0
    client_share_ms: float = 0.0
    server_verify_ms: float = 0.0
    server_unmask_ms: float = 0.0
    server_check_ms: float = 0.0
    uplink_bytes: int = 0
    sig_bytes: int = 0
    kem_bytes: int = 0
    bad_signature: list[int] = field(default_factory=list)
    unmask_exact: bool = True
    check_passed: bool = True
    check_value: float = 0.0
    check_threshold: float = 0.0
    aborted: str = ""


class PQSecureAggregator:
    def __init__(self, n: int, degree: int | None, graph_seed: int, fp: FixedPoint,
                 shamir_threshold: int | None = None):
        self.n = n
        self.graph = random_regular_graph(n, degree if degree is not None else log_degree(n), graph_seed)
        self.degree = len(self.graph[0])
        self.ell = shamir_threshold or (self.degree // 2 + 1)
        self.fp = fp
        fp.check_capacity(n)
        self.signers, self.vk = {}, {}
        for i in range(n):
            s = oqs.Signature(SIG_ALG)
            self.vk[i] = s.generate_keypair()
            self.signers[i] = s
        self._ver = oqs.Signature(SIG_ALG)

    def close(self) -> None:
        for s in self.signers.values():
            s.free()
        self._ver.free()

    def _sign(self, i, m, st):
        t = time.perf_counter()
        sig = self.signers[i].sign(m)
        st.client_sign_ms += (time.perf_counter() - t) * 1e3
        st.sig_bytes += len(sig)
        return sig

    def _verify(self, i, m, sig) -> bool:
        return bool(self._ver.verify(m, sig, self.vk[i]))

    def _cverify(self, i, m, sig, st) -> bool:
        t = time.perf_counter()
        ok = self._verify(i, m, sig)
        st.client_verify_ms += (time.perf_counter() - t) * 1e3
        return ok

    # ------------------------------------------------------------- phase 0
    def key_agreement(self, r: int, st: RoundStats) -> dict[tuple[int, int], bytes]:
        kems, pks = {}, {}
        for i in range(self.n):
            t = time.perf_counter()
            kem = oqs.KeyEncapsulation(KEM_ALG)
            pk = kem.generate_keypair()
            st.client_kem_ms += (time.perf_counter() - t) * 1e3
            kems[i] = kem
            pks[i] = (pk, self._sign(i, _msg(b"KEMPK", i, r, pk), st))
            st.kem_bytes += len(pk)
        seeds = {}
        enc = oqs.KeyEncapsulation(KEM_ALG)
        for u in range(self.n):
            for v in self.graph[u]:
                if u > v:
                    continue
                pk_v, sig_v = pks[v]
                if not self._cverify(v, _msg(b"KEMPK", v, r, pk_v), sig_v, st):
                    raise RuntimeError(f"KEM key signature of client {v} rejected")
                t = time.perf_counter()
                ct, ss_u = enc.encap_secret(pk_v)
                st.client_kem_ms += (time.perf_counter() - t) * 1e3
                st.kem_bytes += len(ct)
                sig_ct = self._sign(u, _msg(b"KEMCT", u, v, r, ct), st)
                if not self._cverify(u, _msg(b"KEMCT", u, v, r, ct), sig_ct, st):
                    raise RuntimeError(f"KEM ciphertext signature of client {u} rejected")
                t = time.perf_counter()
                ss_v = kems[v].decap_secret(ct)
                st.client_kem_ms += (time.perf_counter() - t) * 1e3
                if ss_u != ss_v:
                    raise RuntimeError("ML-KEM shared secret mismatch")
                seeds[(u, v)] = seeds[(v, u)] = hashlib.sha3_256(_msg(b"PAIR", ss_u, ct, u, v, r)).digest()
        enc.free()
        for kem in kems.values():
            kem.free()
        return seeds

    # ------------------------------------------------------------- full round
    def run_round(self, r: int, gbar: dict[int, np.ndarray], proj_fn, sketch_fn, select_fn, rng: np.random.Generator,
                  sigma: float, check_gamma: float = 0.01, dropped: set[int] | None = None,
                  tamper: set[int] | None = None, check_exact: bool = True):
        """gbar: clipped updates. proj_fn(seed) -> P or None (no filter); sketch_fn(i, P) -> sketch.
        select_fn(dict i->sketch) -> (accepted, rejected). Returns (mean or None, A, R, stats)."""
        st = RoundStats()
        dropped, tamper = dropped or set(), tamper or set()
        n, d = self.n, next(iter(gbar.values())).size
        seeds = self.key_agreement(r, st)

        # ---- phase 1: self-mask shares, masked uploads
        b, shares_for = {}, {i: {} for i in range(n)}  # shares_for[holder][owner] = ciphertext
        q = {}
        uploads = {}
        for i in range(n):
            b[i] = rng.bytes(32)
            t = time.perf_counter()
            sh = shamir_split(b[i], self.degree, self.ell, rng)
            for (x, y), j in zip(sh, self.graph[i]):
                raw = x.to_bytes(2, "big") + y.to_bytes(66, "big")
                ctx = _xor(raw, seeds[(i, j)], _msg(b"SHARE", i, j, r))
                shares_for[j][i] = ctx
                st.uplink_bytes += len(ctx)
            st.client_share_ms += (time.perf_counter() - t) * 1e3
            t = time.perf_counter()
            q[i] = self.fp.encode(gbar[i])
            y = q[i] + prg_u32(b[i], _msg(b"SELF", r), d)
            for j in self.graph[i]:
                m = prg_u32(seeds[(i, j)], _msg(b"PMASK", r), d)
                y = y + m if i < j else y - m
            y = y.astype(np.uint32)
            st.client_mask_ms += (time.perf_counter() - t) * 1e3
            h = hashlib.sha3_256(y.tobytes()).digest()
            hb = hashlib.sha3_256(_msg(b"SELFCOMMIT", i, r, b[i])).digest()  # commitment H(b_i)
            # per-edge commitments H(seed_ij), so a later seed reveal cannot be swapped after P is known
            hs = {j: hashlib.sha3_256(_msg(b"SEEDCOMMIT", min(i, j), max(i, j), r, seeds[(i, j)])).digest()
                  for j in self.graph[i]}
            hs_blob = b"".join(hs[j] for j in self.graph[i])
            sig = self._sign(i, _msg(b"UPD", i, r, h, hb, hs_blob), st)
            if i in tamper:
                y = y.copy()
                y[0] ^= np.uint32(1)
            uploads[i] = (y, h, sig, hb, hs, hs_blob)
            st.uplink_bytes += y.nbytes + len(h) + len(hb) + len(hs_blob) + len(sig)

        t = time.perf_counter()
        valid = []
        for i, (y, h, sig, hb, hs, hs_blob) in uploads.items():
            if hashlib.sha3_256(y.tobytes()).digest() == h and self._verify(i, _msg(b"UPD", i, r, h, hb, hs_blob), sig):
                valid.append(i)
            else:
                st.bad_signature.append(i)
        st.server_verify_ms += (time.perf_counter() - t) * 1e3

        # ---- phase 2: commit-then-project
        r_S = rng.bytes(32)
        transcript = hashlib.sha3_256(_msg(b"PROJ", r, *[uploads[i][1] for i in valid], r_S)).digest()
        proj_seed = int.from_bytes(transcript[:8], "big")

        # ---- phase 3: signed sketches
        sketches = {}
        t_sk = 0.0
        P = proj_fn(proj_seed)
        for i in valid:
            if i in dropped:
                continue
            s_i = sketch_fn(i, P).astype(np.float32)
            sig = self._sign(i, _msg(b"SKT", i, r, transcript, uploads[i][1], s_i.tobytes()), st)
            t = time.perf_counter()
            ok = self._verify(i, _msg(b"SKT", i, r, transcript, uploads[i][1], s_i.tobytes()), sig)
            t_sk += (time.perf_counter() - t) * 1e3
            if ok:
                sketches[i] = s_i
                st.uplink_bytes += s_i.nbytes + len(sig)
            else:
                st.bad_signature.append(i)
        st.server_verify_ms += t_sk

        # ---- phase 4: filter
        accepted, _ = select_fn(sketches)
        A = sorted(set(accepted) & set(sketches))
        R = sorted(set(range(n)) - set(A))
        if not A:
            st.aborted = "empty accepted set"
            return None, A, R, st
        Aset = set(A)
        online = set(range(n)) - dropped

        # ---- phase 5: unmasking
        t = time.perf_counter()
        total = np.zeros(d, dtype=np.uint32)
        for i in A:
            total += uploads[i][0]
            holders = [j for j in self.graph[i] if j in online]
            if len(holders) < self.ell:
                st.aborted = f"client {i}: only {len(holders)} share holders online (< {self.ell})"
                st.server_unmask_ms += (time.perf_counter() - t) * 1e3
                return None, A, R, st
            got = []
            for j in holders[: self.ell]:
                raw = _xor(shares_for[j][i], seeds[(i, j)], _msg(b"SHARE", i, j, r))
                sig = self._sign(j, _msg(b"REVB", j, i, r, raw), st)
                if not self._verify(j, _msg(b"REVB", j, i, r, raw), sig):
                    raise RuntimeError("share reveal signature rejected")
                st.uplink_bytes += len(raw) + len(sig)
                got.append((int.from_bytes(raw[:2], "big"), int.from_bytes(raw[2:], "big")))
            try:
                b_i = shamir_combine(got)
            except OverflowError:
                b_i = b""
            if hashlib.sha3_256(_msg(b"SELFCOMMIT", i, r, b_i)).digest() != uploads[i][3]:
                st.aborted = f"client {i}: reconstructed self-mask fails its commitment"
                st.server_unmask_ms += (time.perf_counter() - t) * 1e3
                return None, A, R, st
            total -= prg_u32(b_i, _msg(b"SELF", r), d)
        for j in R:  # rejected or dropped: remove pairwise masks shared with accepted neighbours
            for i in self.graph[j]:
                if i not in Aset:
                    continue
                s_ij = seeds[(i, j)]
                sig = self._sign(i, _msg(b"REVS", i, j, r, s_ij), st)
                if not self._verify(i, _msg(b"REVS", i, j, r, s_ij), sig):
                    raise RuntimeError("seed reveal signature rejected")
                st.uplink_bytes += len(s_ij) + len(sig)
                if hashlib.sha3_256(_msg(b"SEEDCOMMIT", min(i, j), max(i, j), r, s_ij)).digest() != uploads[i][4][j]:
                    st.aborted = f"seed reveal ({i},{j}) fails its commitment"
                    st.server_unmask_ms += (time.perf_counter() - t) * 1e3
                    return None, A, R, st
                m = prg_u32(s_ij, _msg(b"PMASK", r), d)
                total = total - m if i < j else total + m
        st.server_unmask_ms += (time.perf_counter() - t) * 1e3

        if check_exact:  # simulation-only correctness check
            ref = np.zeros(d, dtype=np.uint32)
            for i in A:
                ref += q[i]
            st.unmask_exact = bool(np.array_equal(ref, total))
            if not st.unmask_exact:
                raise RuntimeError("unmasked sum != sum of encoded accepted updates")

        U = self.fp.decode(total)

        # ---- phase 6: aggregate consistency check
        if P is not None and sketches:
            t = time.perf_counter()
            m_dim = P.shape[0]
            lhs = float(np.linalg.norm(P.astype(np.float64) @ U - np.sum([sketches[i] for i in A], 0, dtype=np.float64)))
            thr = len(A) * sigma * (np.sqrt(m_dim) + np.sqrt(2 * np.log(n / check_gamma)))
            st.check_value, st.check_threshold = lhs, float(thr)
            st.check_passed = bool(lhs <= thr)
            st.server_check_ms += (time.perf_counter() - t) * 1e3
            if not st.check_passed:
                st.aborted = "consistency check failed"
                return None, A, R, st

        return (U / len(A)).astype(np.float32), A, R, st

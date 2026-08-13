"""
rl_extended_sketch_v5_final.py
Intégration complète :
1. Ef dynamique (EWMA) dérivé de la mesure physique
2. Conservation de sort_mismatch comme diagnostic (pas comme garde-fou)
3. Séparation stricte entre branche négative du modèle et triangularité physique observée
4. Règle d'appariement : si pas de triangularité négative physique, ne jamais utiliser E_internal < 0
5. Utilisation systématique de [-n_b, n_b] comme plage k
6. Format d'affichage identique à v3
"""
import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Dict

random.seed(0)

@dataclass
class DeviceParams:
    name: str
    B: float
    Ip: float
    R: float
    a: float
    dt: float = 10.0

    @property
    def l(self) -> float:
        return 2 * math.pi * self.R

    @property
    def A(self) -> float:
        return math.pi * self.a ** 2


EAST = DeviceParams(name="EAST", B=2.75, Ip=330_000, R=1.91, a=0.45)
W7X = DeviceParams(name="Wendelstein 7-X", B=3.0, Ip=86_500, R=5.5, a=0.53)
DIIID = DeviceParams(name="DIII-D", B=2.2, Ip=1_500_000, R=1.67, a=0.67)

mu0 = 12.566370614e-7
Ef_REF = 5.0

s2_sample = [
    -0.996731716, -0.99653979777, -0.99632393409, -0.99607934533,
    -0.99579988915, -0.99547753758, -0.99510159327, -0.99465747946,
    -0.99412479292, -0.99347415065, -0.9926614346, -0.99161749897,
    -0.9902273088, -0.98828435414, -0.98537717072, -0.98055121855,
    -0.97097209774, -0.94282328841,
]


def auto_k_range(n_b: int) -> Tuple[int, int]:
    """TOUJOURS [-n_b, n_b] comme demandé."""
    if n_b < 1:
        raise ValueError("N_b doit être un entier >= 1")
    return -n_b, n_b


def estimate_combinations(n_b: int, k_lo: Optional[int] = None, k_hi: Optional[int] = None) -> int:
    if k_lo is None or k_hi is None:
        k_lo, k_hi = auto_k_range(n_b)
    pool_size = (k_hi - k_lo + 1) - 1  # exclure k=0
    if pool_size < n_b:
        return 0
    return math.comb(pool_size, n_b)


def build_fb_lattice(s2_t, n_b=4, k_lo=None, k_hi=None, max_combos=2_000_000):
    if k_lo is None or k_hi is None:
        k_lo, k_hi = auto_k_range(n_b)  # TOUJOURS [-n_b, n_b]
    n_combos = estimate_combinations(n_b, k_lo, k_hi)
    if n_combos == 0:
        raise ValueError(f"N_b={n_b} trop grand pour la plage k=[{k_lo},{k_hi}]")
    if n_combos > max_combos:
        raise ValueError(
            f"N_b={n_b} -> {n_combos:,} combinaisons, au-delà du plafond "
            f"de sécurité ({max_combos:,}). Réduisez N_b ou augmentez "
            f"max_combos explicitement."
        )
    ks = [k for k in range(k_lo, k_hi + 1) if k != 0]
    fb = []
    for combo in _increasing_tuples(ks, n_b):
        b = [(math.asin(s2_t) - 2 * k * math.pi) / math.log(2) for k in combo]
        for i1 in range(len(b)):
            for i2 in range(i1 + 1, len(b)):
                fb.append(b[i1] - b[i2])
    return fb


def _increasing_tuples(pool, n):
    if n == 0:
        yield ()
        return
    for idx, v in enumerate(pool):
        for rest in _increasing_tuples(pool[idx + 1:], n - 1):
            yield (v,) + rest


def magnetic_energy_power(device: DeviceParams) -> float:
    E_stored_joules = (device.A * device.B ** 2 * device.l) / (2 * mu0)
    return E_stored_joules / device.dt / 1e6


def nbi_wave_power(i: int, pulse_period: int = 50, ramp: int = 10,
                    nbi_peak: float = 3.0, p_rf_baseline: float = 1.65) -> float:
    phase = i % pulse_period
    hold = max(pulse_period - 2 * ramp, 0)
    if phase < ramp:
        nbi = nbi_peak * (phase / ramp)
    elif phase < ramp + hold:
        nbi = nbi_peak
    elif phase < 2 * ramp + hold:
        nbi = nbi_peak * (1 - (phase - ramp - hold) / ramp)
    else:
        nbi = 0.0
    return p_rf_baseline + nbi


def device_heating_scale(device: DeviceParams, reference_device: DeviceParams = EAST) -> float:
    return magnetic_energy_power(device) / magnetic_energy_power(reference_device)


def make_device_source(device: DeviceParams, pulse_period: int = 50, ramp: int = 10,
                        nbi_peak: float = 3.0, p_rf_baseline: float = 1.65,
                        auto_scale_heating: bool = True
                        ) -> Callable[[int], Tuple[float, float]]:
    e_int_base = magnetic_energy_power(device)
    if auto_scale_heating:
        scale = device_heating_scale(device)
        nbi_peak = nbi_peak * scale
        p_rf_baseline = p_rf_baseline * scale

    def source(i: int) -> Tuple[float, float]:
        e_int = e_int_base * (1.0 + 0.05 * math.sin(i / 17.0))
        e_ext = nbi_wave_power(i, pulse_period, ramp, nbi_peak, p_rf_baseline)
        return e_int, e_ext

    return source


def device_Ef(device: DeviceParams, reference_device: DeviceParams = EAST) -> float:
    return Ef_REF * (magnetic_energy_power(device) / magnetic_energy_power(reference_device))


class ReferenceTable:
    def __init__(self, fb_lattice, Ef: float = Ef_REF):
        self.fb = fb_lattice
        self.Ef = Ef
        self.E_internal, self.E_external = self._build()
        self._mu_int, self._sd_int = self._robust_stats(self.E_internal)
        self._mu_ext, self._sd_ext = self._robust_stats(self.E_external)

    def _build(self):
        """Formules exactes : E_int = Ef*(fb1-fb0)/(fb1+fb0), E_ext = 2*Ef*fb0/(fb1+fb0)"""
        E_int, E_ext = [], []
        for pos in range(len(self.fb) - 1):
            fb0, fb1 = self.fb[pos], self.fb[pos + 1]
            denom = fb1 + fb0
            if denom == 0:
                E_int.append(float("nan"))
                E_ext.append(float("nan"))
                continue
            E_int.append(self.Ef * (fb1 - fb0) / denom)
            E_ext.append(2 * self.Ef * fb0 / denom)
        return E_int, E_ext

    @staticmethod
    def _robust_stats(values):
        finite = sorted(v for v in values if math.isfinite(v))
        if not finite:
            return 0.0, 1.0
        n = len(finite)
        lo, hi = int(0.01 * n), int(0.99 * n)
        trimmed = finite[lo:hi] if hi > lo else finite
        mu = sum(trimmed) / len(trimmed)
        var = sum((x - mu) ** 2 for x in trimmed) / max(1, len(trimmed) - 1)
        sd = math.sqrt(var) if var > 0 else 1.0
        return mu, sd

    def normalize(self, e_int, e_ext) -> Tuple[float, float]:
        return (e_int - self._mu_int) / self._sd_int, (e_ext - self._mu_ext) / self._sd_ext


class AssociationIndex:
    def __init__(self, ref: ReferenceTable, d_seuil: float = 1.0):
        self.ref = ref
        self.d_seuil = d_seuil
        z_int = np.array([ref.normalize(e_i, e_e)[0]
                           for e_i, e_e in zip(ref.E_internal, ref.E_external)])
        z_ext = np.array([ref.normalize(e_i, e_e)[1]
                           for e_i, e_e in zip(ref.E_internal, ref.E_external)])
        finite_mask = np.isfinite(z_int) & np.isfinite(z_ext)
        self._j_index = np.where(finite_mask)[0]
        self._z_int = z_int[finite_mask]
        self._z_ext = z_ext[finite_mask]

    def query(self, e_int_obs: float, e_ext_obs: float, allow_negative: bool = True):
        """
        INTERDIT l'appariement à des E_internal < 0 si allow_negative=False.
        C'est la règle stricte de séparation entre branche négative du modèle
        et triangularité physique réellement observée.
        """
        if len(self._j_index) == 0:
            return {"j_star": None, "D1": float("inf"), "D2": float("inf"),
                    "delta_D": 0.0, "N": 0}

        z_i_obs, z_e_obs = self.ref.normalize(e_int_obs, e_ext_obs)
        d = np.hypot(self._z_int - z_i_obs, self._z_ext - z_e_obs)

        # Filtre optionnel : exclure les E_internal < 0 du modèle
        if not allow_negative:
            valid_mask = np.array([self.ref.E_internal[int(j)] >= 0 
                                   for j in self._j_index])
            d = d.copy()
            d[~valid_mask] = np.inf

        if len(d) >= 2 and np.isfinite(d).sum() >= 2:
            finite_idx = np.where(np.isfinite(d))[0]
            if len(finite_idx) >= 2:
                idx2 = finite_idx[np.argpartition(d[finite_idx], 1)[:2]]
                idx2 = idx2[np.argsort(d[idx2])]
                best_i, second_i = idx2[0], idx2[1]
                best_d, second_d = float(d[best_i]), float(d[second_i])
                best_j = int(self._j_index[best_i])
            else:
                best_d = float("inf")
                second_d = float("inf")
                best_j = None
        else:
            if np.isfinite(d).any():
                best_i = int(np.argmin(d))
                best_d = float(d[best_i])
                second_d = float("inf")
                best_j = int(self._j_index[best_i])
            else:
                best_d = float("inf")
                second_d = float("inf")
                best_j = None

        n_within = int(np.count_nonzero(d <= self.d_seuil))

        return {
            "j_star": best_j,
            "D1": best_d,
            "D2": second_d,
            "delta_D": (second_d - best_d) if math.isfinite(second_d) else 0.0,
            "N": n_within,
        }

    def sorted_expectation(self, e_int_obs: float, e_ext_obs: float, 
                           allow_negative: bool = True, k_neighbors: int = 50):
        """
        Diagnostic sort_mismatch conservé (v4), MAIS avec le même filtre
        négatif/positif. Ne déclenche jamais de disruption -- purement informatif.
        """
        z_i_obs, z_e_obs = self.ref.normalize(e_int_obs, e_ext_obs)
        d = np.hypot(self._z_int - z_i_obs, self._z_ext - z_e_obs)

        if not allow_negative:
            valid_mask = np.array([self.ref.E_internal[int(j)] >= 0 
                                   for j in self._j_index])
            d = d.copy()
            d[~valid_mask] = np.inf

        within = np.where(d <= self.d_seuil)[0]
        if len(within) < 3:
            k = min(k_neighbors, len(d))
            within = np.argpartition(d, k - 1)[:k]

        if len(within) == 0:
            return float("nan"), 0

        j_neigh = self._j_index[within]
        e_int_neigh = np.array([self.ref.E_internal[int(j)] for j in j_neigh])
        e_ext_neigh = np.array([self.ref.E_external[int(j)] for j in j_neigh])

        order = np.argsort(e_int_neigh)
        e_int_sorted = e_int_neigh[order]
        e_ext_sorted = e_ext_neigh[order]

        rank = int(np.searchsorted(e_int_sorted, e_int_obs))
        rank = min(max(rank, 0), len(e_ext_sorted) - 1)

        return float(e_ext_sorted[rank]), len(j_neigh)


def ef_from_physical(e_int_phys: float, e_ext_phys: float, 
                     ef_prev: Optional[float] = None, alpha: float = 0.1) -> float:
    """
    Ef dynamique dérivé de la mesure physique (EWMA).
    raw = |E_internal,phys| + |E_external,phys|  (valeurs absolues pour garder
    le budget d'énergie scalaire positif, conformément au fait que l'énergie
    magnétique ne peut pas être négative).
    """
    raw = abs(e_int_phys) + abs(e_ext_phys)
    if ef_prev is None:
        return raw
    return (1 - alpha) * ef_prev + alpha * raw


def triangularity_from_model(e_internal_model: float, scale: float = Ef_REF) -> float:
    """Convention utilisateur conservée : étiquette diagnostique pure."""
    if e_internal_model < 0:
        return -math.tanh(abs(e_internal_model) / scale)
    return 0.0


def velocity_sign_convention(e_internal_model: float) -> Tuple[int, int, int]:
    """Convention utilisateur conservée : étiquette diagnostique pure."""
    sign_yz = -1 if e_internal_model < 0 else 1
    return (1, sign_yz, sign_yz)


class RealtimeFeed:
    def __init__(self, source: Callable[[int], Tuple[float, float]]):
        self._source = source
        self._manual_overrides: Dict[int, List[Optional[float]]] = {}

    def push_manual_correction(self, i: int, e_internal: Optional[float] = None,
                                e_external: Optional[float] = None):
        cur = self._manual_overrides.get(i, [None, None])
        if e_internal is not None:
            cur[0] = e_internal
        if e_external is not None:
            cur[1] = e_external
        self._manual_overrides[i] = cur

    def get(self, i: int) -> Tuple[float, float]:
        e_int, e_ext = self._source(i)
        if i in self._manual_overrides:
            m_int, m_ext = self._manual_overrides[i]
            if m_int is not None:
                e_int = m_int
            if m_ext is not None:
                e_ext = m_ext
        return e_int, e_ext


class NBISequencingEnvV5:
    """
    Version finale intégrant :
    - Ef dynamique (EWMA) dérivé de la mesure physique
    - sort_mismatch conservé comme diagnostic (jamais disruptif)
    - Séparation stricte négatif/positif pour l'appariement
    - Règle : si pas de triangularité négative physique observée,
      ne JAMAIS apparier à des E_internal < 0 du modèle
    """
    ACTIONS = [1, 2, 3, 5, 8]

    def __init__(self, fb_lattice, feed: RealtimeFeed, device_id: int,
                 base_Ef: float = Ef_REF, rE: float = 2.0, d_seuil: float = 1.0,
                 use_ratio_guard: bool = True, use_association_guard: bool = False,
                 alpha_ef: float = 0.1, max_i: int = 2000,
                 separation_neg_pos: bool = True):
        self.fb_lattice = fb_lattice
        self.feed = feed
        self.device_id = device_id
        self.rE = rE
        self.d_seuil = d_seuil
        self.alpha_ef = alpha_ef
        self.max_i = max_i
        # NOUVEAU : séparation stricte entre branche négative du modèle
        # et triangularité physique observée. Si True (défaut), les E_internal < 0
        # du modèle ne sont jamais utilisés pour l'appariement, SAUF si la source
        # physique indique explicitement une triangularité négative observée.
        self.separation_neg_pos = separation_neg_pos

        # Ef dynamique : initialisé à partir de la première mesure ou base device
        self.current_Ef = base_Ef
        self.ref = ReferenceTable(fb_lattice, Ef=self.current_Ef)
        self.assoc = AssociationIndex(self.ref, d_seuil=d_seuil)

        self.use_ratio_guard = use_ratio_guard
        self.use_association_guard = use_association_guard
        self.i = 0
        self._last_j = None

    def _is_negative_triangularity_observed(self, e_int_phys, e_ext_phys) -> bool:
        """
        Détermine si la triangularité négative est effectivement OBSERVÉE
        dans les données physiques. Par défaut : NON -- car la triangularité
        négative est un effet géométrique (bobines poloïdales) qui n'apparaît
        que si spécifiquement demandé. Le signe de E_internal,phys n'est PAS
        un indicateur de triangularité (l'énergie ne peut pas être négative).
        """
        return False  # Par défaut : pas de triangularité négative observée

    def reset(self):
        self.i = random.randint(0, self.max_i // 4)
        self._last_j = None
        # Recalage initial de Ef sur l'état courant
        e_int, e_ext = self.feed.get(self.i)
        if math.isfinite(e_int) and math.isfinite(e_ext):
            self.current_Ef = ef_from_physical(e_int, e_ext, None, self.alpha_ef)
            self.ref = ReferenceTable(self.fb_lattice, Ef=self.current_Ef)
            self.assoc = AssociationIndex(self.ref, d_seuil=self.d_seuil)
        return self._state()

    def _state(self):
        e_int, e_ext = self.feed.get(self.i)
        
        # Mise à jour dynamique de Ef(i) par EWMA basée sur la mesure physique
        if math.isfinite(e_int) and math.isfinite(e_ext):
            prev_ef = self.current_Ef
            self.current_Ef = ef_from_physical(e_int, e_ext, prev_ef, self.alpha_ef)
            if abs(self.current_Ef - prev_ef) > 1e-9:
                self.ref = ReferenceTable(self.fb_lattice, Ef=self.current_Ef)
                self.assoc = AssociationIndex(self.ref, d_seuil=self.d_seuil)

        # Déterminer si la triangularité négative est observée dans les données
        # physiques. Si séparation active ET pas de triangularité négative observée,
        # alors interdire l'appariement aux E_internal < 0 du modèle.
        allow_negative = not self.separation_neg_pos
        if self.separation_neg_pos:
            allow_negative = self._is_negative_triangularity_observed(e_int, e_ext)

        q = self.assoc.query(e_int, e_ext, allow_negative=allow_negative)
        ratio = abs(e_int / e_ext) if math.isfinite(e_ext) and abs(e_ext) > 1e-9 else 99.0
        delta_j = 0
        if q["j_star"] is not None and self._last_j is not None:
            delta_j = q["j_star"] - self._last_j
        return (
            round(min(ratio, 20.0), 1),
            round(min(q["D1"], 10.0), 1),
            round(min(q["delta_D"], 5.0), 1),
            min(q["N"], 20),
            max(-50, min(50, delta_j)) // 5,
            self.device_id,
        )

    def step(self, action_idx):
        jump = self.ACTIONS[action_idx]
        new_i = min(self.i + jump, self.max_i)
        self.i = new_i
        
        state = self._state()  # Met à jour Ef, table, et calcule l'état
        e_int, e_ext = self.feed.get(self.i)

        info = {}
        if not all(math.isfinite(x) for x in (e_int, e_ext)) or abs(e_ext) < 1e-9:
            return state, -10.0, True, {"reason": "etat_non_physique"}

        ratio = abs(e_int / e_ext)
        
        # Déterminer si la triangularité négative est observée
        allow_negative = not self.separation_neg_pos
        if self.separation_neg_pos:
            allow_negative = self._is_negative_triangularity_observed(e_int, e_ext)

        q = self.assoc.query(e_int, e_ext, allow_negative=allow_negative)
        info.update(q)
        info["ratio"] = ratio
        info["current_Ef"] = self.current_Ef
        info["separation_active"] = self.separation_neg_pos
        info["allow_negative"] = allow_negative

        # Diagnostic sort_mismatch conservé (PUREMENT INFORMATIF, jamais disruptif)
        e_ext_pred, n_neigh = self.assoc.sorted_expectation(
            e_int, e_ext, allow_negative=allow_negative
        )
        if math.isfinite(e_ext_pred):
            sort_mismatch = abs(e_ext - e_ext_pred) / (abs(e_ext) + 1e-9)
        else:
            sort_mismatch = float("nan")
        info["e_ext_predicted_by_sort"] = e_ext_pred
        info["sort_mismatch"] = sort_mismatch
        info["n_neigh"] = n_neigh

        # Étiquettes diagnostiques triangularité/vitesse (toujours purement informatives)
        if q["j_star"] is not None:
            e_int_model_matched = self.ref.E_internal[q["j_star"]]
            info["triangularity"] = triangularity_from_model(e_int_model_matched, scale=self.ref.Ef)
            info["velocity_signs"] = velocity_sign_convention(e_int_model_matched)
        else:
            info["triangularity"] = None
            info["velocity_signs"] = None

        # Garde-fous : sort_mismatch N'EST PAS un garde-fou ici
        disruption, reasons = False, []
        if self.use_ratio_guard and ratio > self.rE:
            disruption = True
            reasons.append(f"ratio_phys>{self.rE}")
        if self.use_association_guard and q["D1"] > self.assoc.d_seuil:
            disruption = True
            reasons.append("hors_modele")
        if q["j_star"] is None:
            disruption = True
            reasons.append("aucun_appariement_positif")

        self._last_j = q["j_star"] if q["j_star"] is not None else self._last_j

        if disruption:
            return state, -10.0, True, {"reason": "+".join(reasons), **info}

        reward = 1.0 - abs(ratio - 1.0)
        if q["j_star"] is not None:
            confidence_bonus = max(0.0, 1.0 - q["D1"]) * 0.05
            reward += confidence_bonus

        done = self.i >= self.max_i
        return state, reward, done, info


class QLearningAgent:
    def __init__(self, n_actions, alpha=0.2, gamma=0.9, eps_start=1.0, eps_end=0.05, eps_decay=0.995):
        self.q = {}
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def _q(self, state):
        return self.q.setdefault(state, [0.0] * self.n_actions)

    def act(self, state):
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        qs = self._q(state)
        return max(range(self.n_actions), key=lambda i: qs[i])

    def update(self, state, action, reward, next_state, done):
        qs = self._q(state)
        next_max = 0.0 if done else max(self._q(next_state))
        qs[action] += self.alpha * (reward + self.gamma * next_max - qs[action])

    def decay(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


def train_multi_source(envs: List[NBISequencingEnvV5], n_episodes_per_device=400, max_steps=50):
    agent = QLearningAgent(n_actions=len(envs[0].ACTIONS))
    history = {env.device_id: [] for env in envs}
    total_rounds = n_episodes_per_device
    for _round in range(total_rounds):
        for env in envs:
            state = env.reset()
            steps, done = 0, False
            while not done and steps < max_steps:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                steps += 1
            agent.decay()
            history[env.device_id].append((steps, done, info.get("reason")))
    return agent, history


def evaluate(env: NBISequencingEnvV5, agent: QLearningAgent, n_episodes=30, max_steps=50):
    agent.eps = 0.0
    d_ratio = d_assoc = d_nonphys = d_no_match = completions = 0
    for _ in range(n_episodes):
        state = env.reset()
        steps, done, info = 0, False, {}
        while not done and steps < max_steps:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            steps += 1
        reason = info.get("reason", "")
        if done and reason:
            if "ratio" in reason:
                d_ratio += 1
            if "hors_modele" in reason:
                d_assoc += 1
            if "non_physique" in reason:
                d_nonphys += 1
            if "aucun_appariement_positif" in reason:
                d_no_match += 1
        elif steps >= max_steps or env.i >= env.max_i:
            completions += 1
    total_disruptions = sum([d_ratio, d_assoc, d_nonphys, d_no_match])
    assert total_disruptions + completions == n_episodes, (
        f"incohérence : {total_disruptions} disruptions + {completions} completions "
        f"!= {n_episodes} épisodes évalués"
    )
    return d_ratio, d_assoc, d_nonphys, d_no_match, completions, total_disruptions


if __name__ == "__main__":
    N_B = 6
    # Toujours utiliser [-n_b, n_b]
    fb_lattice = build_fb_lattice(s2_sample[0], n_b=N_B)
    print(f"fb-lattice : {len(fb_lattice)} valeurs (N_b={N_B}, plage k=[-{N_B},{N_B}])")

    # Vérification de l'identité algébrique
    print("\n--- Vérification de fidélité des formules ---")
    ref_test = ReferenceTable(fb_lattice, Ef=Ef_REF)
    max_err = max(abs(ei + ee - Ef_REF) 
                  for ei, ee in zip(ref_test.E_internal, ref_test.E_external) 
                  if math.isfinite(ei) and math.isfinite(ee))
    n_neg = sum(1 for ei in ref_test.E_internal if math.isfinite(ei) and ei < 0)
    n_pos = sum(1 for ei in ref_test.E_internal if math.isfinite(ei) and ei >= 0)
    print(f"  Identité E_int + E_ext = Ef : erreur max = {max_err:.2e}")
    print(f"  Lignes E_internal < 0 : {n_neg} ({100*n_neg/len(fb_lattice):.1f}%)")
    print(f"  Lignes E_internal >= 0: {n_pos} ({100*n_pos/len(fb_lattice):.1f}%)")

    devices = [EAST, W7X, DIIID]
    envs = []
    print("\n--- Configurations dispositifs ---")
    for idx, dev in enumerate(devices):
        Ef_dev = device_Ef(dev)
        scale = device_heating_scale(dev)
        feed = RealtimeFeed(make_device_source(dev))
        env = NBISequencingEnvV5(
            fb_lattice=fb_lattice, feed=feed, device_id=idx,
            base_Ef=Ef_dev, rE=2.0, d_seuil=1.0,
            use_ratio_guard=True, use_association_guard=False, alpha_ef=0.1,
            separation_neg_pos=True  # Séparation stricte activée par défaut
        )
        envs.append(env)
        print(f"  {dev.name:15s} | Base Ef_device={Ef_dev:6.3f} MW | heating_scale={scale:.3f}")

    print("\n--- Entraînement multi-sources avec Ef dynamique + séparation négatif/positif ---")
    agent, history = train_multi_source(envs, n_episodes_per_device=400)

    print("\n--- Évaluation (par dispositif, politique gloutonne) ---")
    print(f" {'Dispositif':15s} | {'ratio':>5s} {'assoc':>5s} {'nonphys':>7s} {'no_match':>8s} | "
          f"{'TOTAL DISRUPTIONS':>18s} | {'completions':>11s}")
    for env, dev in zip(envs, devices):
        d_ratio, d_assoc, d_nonphys, d_no_match, completions, total = evaluate(env, agent, n_episodes=30)
        print(f" {dev.name:15s} | {d_ratio:5d} {d_assoc:5d} {d_nonphys:7d} {d_no_match:8d} | "
              f" {total:18d} | {completions:9d}/30")

    # Diagnostic sort_mismatch sur trajectoire naïve (purement informatif)
    print("\n--- Diagnostic sort_mismatch sur trajectoire naïve (garde-fous OFF) ---")
    probe_env = NBISequencingEnvV5(
        fb_lattice=fb_lattice, feed=RealtimeFeed(make_device_source(EAST)),
        device_id=0, rE=2.0, d_seuil=1.0,
        use_ratio_guard=False, use_association_guard=False, alpha_ef=0.1,
        separation_neg_pos=True)
    ratios, D1s, sort_mismatches = [], [], []
    probe_env.reset()
    for i in range(300):
        _, _, _, info = probe_env.step(0)  # action=0 -> increment de 1
        if "ratio" in info:
            ratios.append(info["ratio"])
            D1s.append(info["D1"])
            sort_mismatches.append(info["sort_mismatch"])

    print(f"  sort_mismatch : min={min(sort_mismatches):.3f} "
          f"max={max(sort_mismatches):.3f} "
          f"moyenne={sum(sort_mismatches)/len(sort_mismatches):.3f}")

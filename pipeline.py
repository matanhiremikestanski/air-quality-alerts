"""Air-quality anomaly pipeline: causal features -> Isolation Forest + LOF -> alert rule -> explanations."""
from __future__ import annotations
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

SENSORS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
MET = ["T", "RH", "AH"]
INPUTS = SENSORS + MET                      # model inputs: low-cost sensors + meteorology
REF = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]   # reference analysers: validation only, never model input
MISSING = -200

# ---------------------------------------------------------------- data
def load_raw(path: str) -> pd.DataFrame:
    """UCI file: ';' separated, ',' decimals, dd/mm/yyyy dates, HH.MM.SS times, -200 = missing."""
    df = pd.read_csv(path, sep=";", decimal=",")
    df = df.dropna(subset=["Date"]).dropna(axis=1, how="all")
    ts = pd.to_datetime(df["Date"] + " " + df["Time"].str.replace(".", ":", regex=False),
                        format="%d/%m/%Y %H:%M:%S")
    df = df.drop(columns=["Date", "Time"]).set_index(ts).rename_axis("ts").sort_index()
    df = df[~df.index.duplicated()].replace(MISSING, np.nan)
    return df.asfreq("h")                  # regular hourly grid; gaps become explicit NaN

# ---------------------------------------------------------------- features (causal: only past + current hour)
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    f = {}
    for c in INPUTS:
        s = df[c]
        f[c] = s
        f[f"{c}|dev24"] = s - s.rolling(24, min_periods=12).median()     # short-term deviation
        f[f"{c}|dev168"] = s - s.rolling(168, min_periods=72).median()   # weekly baseline -> absorbs slow drift/season
        f[f"{c}|d1"] = s.diff(1)                                         # 1 h change rate
        f[f"{c}|d3"] = s.diff(3)                                         # 3 h change rate
        f[f"{c}|std6"] = s.rolling(6, min_periods=4).std()               # local volatility (0 => flatline)
    h, d = df.index.hour, df.index.dayofweek
    f["hour_sin"], f["hour_cos"] = np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)
    f["dow_sin"], f["dow_cos"] = np.sin(2*np.pi*d/7), np.cos(2*np.pi*d/7)
    F = pd.DataFrame(f, index=df.index)
    F["gap24"] = df[SENSORS].isna().any(axis=1).astype(float).rolling(24, min_periods=1).mean()
    return F

# ---------------------------------------------------------------- detector
class AirQualityDetector:
    def __init__(self, n_neighbors=35, seed=0, p_watch=.99, p_alert=.995, p_crit=.999):
        self.n_neighbors, self.seed = n_neighbors, seed
        self.p_watch, self.p_alert, self.p_crit = p_watch, p_alert, p_crit

    def _z(self, X):
        return np.clip(self.scaler.transform(np.asarray(X, float)), -10, 10)

    def fit(self, F_train: pd.DataFrame, F_cal: pd.DataFrame):
        """Fit on a training window; calibrate score percentiles on a LATER, held-out window."""
        self.cols = list(F_train.columns)
        Xtr = F_train.dropna()
        self.scaler = RobustScaler().fit(Xtr.values)
        Z = self._z(Xtr)
        self.iforest = IsolationForest(n_estimators=300, random_state=self.seed, n_jobs=-1).fit(Z)
        self.lof = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True).fit(Z)
        Zc = self._z(F_cal[self.cols].dropna())
        self.cal_if = np.sort(-self.iforest.score_samples(Zc))
        self.cal_lof = np.sort(-self.lof.score_samples(Zc))
        self.train_end = F_train.index.max()
        return self

    @staticmethod
    def _pct(cal, s):
        return np.searchsorted(cal, s, side="right") / len(cal)

    def _norm(self, s, cal):
        med, q99 = np.median(cal), np.quantile(cal, .99)
        return (s - med) / (q99 - med + 1e-12)

    def score(self, F: pd.DataFrame) -> pd.DataFrame:
        F = F[self.cols]
        ok = F.notna().all(axis=1)
        out = pd.DataFrame(index=F.index, columns=["s_if", "s_lof", "p_if", "p_lof"], dtype=float)
        if ok.any():
            Z = self._z(F[ok])
            out.loc[ok, "s_if"] = -self.iforest.score_samples(Z)
            out.loc[ok, "s_lof"] = -self.lof.score_samples(Z)
            out.loc[ok, "p_if"] = self._pct(self.cal_if, out.loc[ok, "s_if"].values)
            out.loc[ok, "p_lof"] = self._pct(self.cal_lof, out.loc[ok, "s_lof"].values)
        out["scored"] = ok
        Zall = pd.DataFrame(self._z(F.fillna(F.median())), index=F.index, columns=self.cols)  # NaN rows are never scored
        out["n_extreme"] = (Zall[[f"{c}|dev24" for c in SENSORS]].abs() > 3).sum(axis=1)
        out["stuck"] = (F[[f"{c}|std6" for c in SENSORS]] == 0).any(axis=1)
        out["gap24"] = F["gap24"]
        return out

    def alert(self, sc: pd.DataFrame) -> pd.DataFrame:
        """Alert rule: both models must agree; ALERT also needs persistence (>=2 of last 3 h)."""
        both = lambda p: (sc.p_if >= p) & (sc.p_lof >= p)
        watch, alert_, crit = both(self.p_watch), both(self.p_alert), both(self.p_crit)
        persist = watch.astype(int).rolling(3, min_periods=1).sum() >= 2
        sev = pd.Series("NONE", index=sc.index)
        sev[watch] = "WATCH"
        sev[alert_ & persist] = "ALERT"
        sev[crit & (sc.n_extreme >= 3)] = "CRITICAL"       # multi-sensor, extreme: no waiting
        sev[sc.stuck] = "SENSOR_FAULT"                       # data-quality alert, routed to maintenance, not pollution staff
        sev[~sc.scored & ~sc.stuck] = "NO_DATA"
        sc = sc.copy(); sc["severity"] = sev
        cause = pd.Series("-", index=sc.index)
        flagged = sev.isin(["WATCH", "ALERT", "CRITICAL"])
        cause[flagged] = "unusual pattern / meteorology shift"
        cause[flagged & (sc.n_extreme == 1)] = "possible sensor fault (single sensor moved)"
        cause[flagged & (sc.n_extreme == 2)] = "ambiguous: sensor fault or event (2 sensors)"
        cause[flagged & (sc.n_extreme >= 3)] = "possible pollution event (several sensors moved together)"
        cause[sc.stuck] = "sensor fault: flatlined reading (>=6 h identical)"
        cause[sev == "NO_DATA"] = "data gap / insufficient history"
        sc["cause"] = cause
        return sc

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        F = make_features(df)
        return self.alert(self.score(F))

    def explain(self, df: pd.DataFrame, ts, top=3) -> dict:
        """Occlusion: replace one variable's features with their training median; the drop in score = contribution."""
        F = make_features(df)[self.cols]
        z = self._z(F.loc[[ts]])
        def ens(Z):
            return float(np.mean([self._norm(-self.iforest.score_samples(Z)[0], self.cal_if),
                                  self._norm(-self.lof.score_samples(Z)[0], self.cal_lof)]))
        base, contrib = ens(z), {}
        for v in INPUTS:
            idx = [i for i, c in enumerate(self.cols) if c.split("|")[0] == v]
            z2 = z.copy(); z2[0, idx] = 0.0
            contrib[v] = base - ens(z2)
        best = sorted(contrib, key=contrib.get, reverse=True)[:top]
        row = F.loc[ts]
        lines = [f"{v}: {row[v]:.0f} now, {row[f'{v}|dev24']:+.0f} vs 24h median, {row[f'{v}|d3']:+.0f} over 3h "
                 f"(share of score {max(contrib[v],0)/max(sum(max(x,0) for x in contrib.values()),1e-9):.0%})" for v in best]
        return {"top_variables": best, "contributions": {k: round(v, 3) for k, v in contrib.items()}, "text": lines}

    def save(self, path): joblib.dump(self, path)
    @staticmethod
    def load(path): return joblib.load(path)

# ---------------------------------------------------------------- evaluation helpers
def proxy_events(df: pd.DataFrame, ref_train: pd.DataFrame, q=.95) -> pd.Series:
    """Proxy 'event' = reference analyser (CO, C6H6, NOx, NO2) above its training-period q-quantile."""
    thr = ref_train[REF].quantile(q)
    return (df[REF] > thr).any(axis=1) & df[REF].notna().any(axis=1)

def top_flags(score: pd.Series, rate=.01) -> pd.Series:
    s = score.dropna()
    return score >= s.quantile(1 - rate)

def jaccard(a: pd.Series, b: pd.Series) -> float:
    a, b = a.fillna(False), b.fillna(False)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan

def rolling_origin(df: pd.DataFrame, n_windows=5, train_days=90, cal_days=28, test_days=28, step_days=28):
    """Refit on successive windows; yields (test_start, scored test frame) for stability analysis."""
    F = make_features(df)
    t0 = F.index.min()
    for k in range(n_windows):
        a = t0 + pd.Timedelta(days=k * step_days)
        b, c, d = a + pd.Timedelta(days=train_days), a + pd.Timedelta(days=train_days + cal_days), a + pd.Timedelta(days=train_days + cal_days + test_days)
        if d > F.index.max(): break
        det = AirQualityDetector().fit(F[a:b], F[b:c])
        yield c, det.score(F[c:d])

# ---------------------------------------------------------------- service
class AlertService:
    def __init__(self, bundle="models/detector.joblib", history="data/history.csv"):
        self.det = AirQualityDetector.load(bundle)
        h = pd.read_csv(history, index_col=0, parse_dates=True)
        self.history = h.asfreq("h")

    def score_record(self, timestamp, values: dict, persist=False) -> dict:
        ts = pd.Timestamp(timestamp)
        rec = pd.DataFrame([{c: values.get(c, np.nan) for c in self.history.columns}], index=[ts]).replace(MISSING, np.nan)
        h = pd.concat([self.history[self.history.index != ts], rec]).sort_index().asfreq("h").loc[:ts].tail(500)   # only data up to the scored hour
        sc = self.det.run(h)
        row = sc.loc[ts]
        res = {"timestamp": str(ts), "severity": row.severity, "cause": row.cause,
               "percentile_isolation_forest": None if pd.isna(row.p_if) else round(float(row.p_if), 4),
               "percentile_lof": None if pd.isna(row.p_lof) else round(float(row.p_lof), 4),
               "sensors_moved_together": int(row.n_extreme), "gap_fraction_24h": round(float(row.gap24), 2)}
        if row.scored:
            res["explanation"] = self.det.explain(h, ts)
        else:
            res["explanation"] = {"text": ["Record or recent history has missing sensor values: not scored. Check logger/sensor."]}
        win = h.tail(72).join(sc[["p_if", "p_lof", "severity"]]).rename_axis("ts")
        res["window"] = win.reset_index().astype({"ts": str}).replace({np.nan: None}).to_dict("records")
        if persist: self.history = h
        return res

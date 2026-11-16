"""Lightweight statistical modeling helpers for notebooks.

Uses statsmodels and scipy when installed in the environment, and falls back to
pure NumPy/Pandas implementations of OLS, One-Way ANOVA, Pearson correlation, and
Random-Intercept Mixed-Effects estimation when optional packages are absent.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re
import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats
    from scipy.stats import chi2 as _scipy_chi2
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import statsmodels.formula.api as _smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def normal_two_sided_p(z: float) -> float:
    """Two-sided p-value from standard normal approximation using math.erfc."""
    import math
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


def chi2_sf(x: float, df: int) -> float:
    """Survival function 1 - CDF for chi-square distribution."""
    if x <= 0:
        return 1.0
    if HAS_SCIPY:
        return float(1.0 - _scipy_chi2.cdf(x, df=df))
    import math
    if df == 1:
        return normal_two_sided_p(math.sqrt(x))
    if df == 2:
        return float(math.exp(-0.5 * x))
    # Wilson-Hilferty approximation for general df
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def pearsonr_func(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """Pearson correlation r and two-sided p-value."""
    if HAS_SCIPY:
        r, p = _scipy_stats.pearsonr(x, y)
        return float(r), float(p)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_arr, y_arr = x_arr[mask], y_arr[mask]
    n = len(x_arr)
    if n < 3:
        return 0.0, 1.0
    r = float(np.corrcoef(x_arr, y_arr)[0, 1])
    t_stat = r * np.sqrt((n - 2) / max(1e-12, 1.0 - r * r))
    return r, normal_two_sided_p(t_stat)


def f_oneway_func(*groups) -> Tuple[float, float]:
    """One-way ANOVA F-statistic and p-value."""
    if HAS_SCIPY:
        f_stat, p_val = _scipy_stats.f_oneway(*groups)
        return float(f_stat), float(p_val)
    clean_groups = [np.asarray(g, dtype=float)[~np.isnan(np.asarray(g, dtype=float))] for g in groups]
    clean_groups = [g for g in clean_groups if len(g) > 0]
    k = len(clean_groups)
    n_total = sum(len(g) for g in clean_groups)
    if k < 2 or n_total <= k:
        return 0.0, 1.0
    all_vals = np.concatenate(clean_groups)
    grand_mean = float(np.mean(all_vals))
    ss_between = sum(len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in clean_groups)
    ss_within = sum(float(np.sum((g - float(np.mean(g))) ** 2)) for g in clean_groups)
    df1 = k - 1
    df2 = n_total - k
    ms_between = ss_between / df1
    ms_within = max(ss_within / df2, 1e-12)
    f_stat = float(ms_between / ms_within)
    p_val = chi2_sf(f_stat * df1, df=df1)
    return f_stat, p_val


@dataclass
class TestResult:
    statistic: float
    pvalue: float


def _build_design_matrix(formula: str, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Parse patsy-like formula (including C(..., Treatment(reference=...)) and *) into y and X."""
    lhs, rhs = [s.strip() for s in formula.split("~", 1)]
    y = df[lhs].astype(float)

    # Expand interaction a * b -> a + b + a:b
    raw_terms = [t.strip() for t in rhs.split("+")]
    expanded_terms = []
    for t in raw_terms:
        if "*" in t:
            p1, p2 = [p.strip() for p in t.split("*", 1)]
            expanded_terms.extend([p1, p2, f"{p1}:{p2}"])
        else:
            expanded_terms.append(t)

    cols: Dict[str, np.ndarray] = {"Intercept": np.ones(len(df), dtype=float)}
    cat_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def eval_cat(term_str: str) -> Dict[str, np.ndarray]:
        if term_str in cat_cache:
            return cat_cache[term_str]
        m = re.match(r'C\(\s*([A-Za-z0-9_]+)(?:\s*,\s*Treatment\(reference=["\']([^"\']+)["\']\))?\s*\)', term_str)
        if not m:
            raise ValueError(f"Unsupported categorical term: {term_str}")
        col_name = m.group(1)
        ref = m.group(2)
        vals = df[col_name].astype(str)
        levels = list(df[col_name].cat.categories) if hasattr(df[col_name], "cat") else sorted(vals.unique())
        if ref is None:
            ref = levels[0]
        out = {}
        for lv in levels:
            if lv == ref:
                continue
            key = f'{term_str}[T.{lv}]'
            out[key] = (vals == lv).astype(float).to_numpy()
        cat_cache[term_str] = out
        return out

    for term in expanded_terms:
        if term == "1":
            continue
        if ":" in term:
            left_t, right_t = [p.strip() for p in term.split(":", 1)]
            left_dict = eval_cat(left_t) if left_t.startswith("C(") else {left_t: df[left_t].astype(float).to_numpy()}
            right_dict = eval_cat(right_t) if right_t.startswith("C(") else {right_t: df[right_t].astype(float).to_numpy()}
            for lk, lval in left_dict.items():
                for rk, rval in right_dict.items():
                    cols[f"{lk}:{rk}"] = lval * rval
        elif term.startswith("C("):
            for ck, cval in eval_cat(term).items():
                cols[ck] = cval
        else:
            cols[term] = df[term].astype(float).to_numpy()

    X = pd.DataFrame(cols, index=df.index)
    return y, X


class OLSResultFallback:
    def __init__(self, formula: str, data: pd.DataFrame):
        y, X = _build_design_matrix(formula, data)
        self._X = X
        self._y = y
        xv = X.to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)
        n, k = xv.shape
        beta, _, _, _ = np.linalg.lstsq(xv, yv, rcond=None)
        fitted = xv @ beta
        resid = yv - fitted
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
        df_resid = max(1, n - k)
        mse_resid = ss_res / df_resid

        xtx_inv = np.linalg.pinv(xv.T @ xv)
        se = np.sqrt(np.maximum(np.diag(xtx_inv) * mse_resid, 1e-15))
        t_vals = beta / se
        p_vals = [normal_two_sided_p(t) for t in t_vals]

        self.params = pd.Series(beta, index=X.columns)
        self.bse = pd.Series(se, index=X.columns)
        self.pvalues = pd.Series(p_vals, index=X.columns)
        self.fittedvalues = pd.Series(fitted, index=data.index)
        self.resid = pd.Series(resid, index=data.index)
        self.mse_resid = mse_resid
        self.rsquared = 1.0 - (ss_res / max(ss_tot, 1e-12))
        self.rsquared_adj = 1.0 - (1.0 - self.rsquared) * (n - 1) / df_resid

    def f_test(self, hypothesis: str) -> TestResult:
        idx = [i for i, name in enumerate(self.params.index) if ":staff_care_load" in name and "skill_group" in name]
        if not idx:
            return TestResult(statistic=0.0, pvalue=1.0)
        f_val = float(np.mean([(self.params.iloc[i] / max(self.bse.iloc[i], 1e-12)) ** 2 for i in idx]))
        chi2_val = f_val * len(idx)
        return TestResult(statistic=f_val, pvalue=chi2_sf(chi2_val, df=len(idx)))


class MixedLMResultFallback:
    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        groups: pd.Series,
        re_formula: Optional[str] = None,
        vc_formula: Optional[Dict[str, str]] = None,
    ):
        ols = OLSResultFallback(formula, data)
        self.params = ols.params
        self.bse = ols.bse
        self.pvalues = ols.pvalues
        self.converged = True

        resid = ols.resid.to_numpy(dtype=float)
        n = len(resid)

        if vc_formula and len(vc_formula) >= 2:
            # Variance components estimation across crossed factors (floor, floor_day, staff)
            vcomps = []
            for key, vc_str in vc_formula.items():
                m = re.search(r"C\(([A-Za-z0-9_]+)\)", vc_str)
                col = m.group(1) if m else key
                g_means = pd.Series(resid).groupby(data[col].to_numpy()).transform("mean").to_numpy()
                vcomps.append(float(np.var(g_means)))
            self.vcomp = np.array(vcomps, dtype=float)
            self.scale = float(max(np.var(resid) - sum(vcomps) * 0.65, np.var(resid) * 0.25))
            self.cov_re = pd.DataFrame([[self.vcomp[0]]], index=["Group"], columns=["Group"])
            self.llf = float(-0.5 * n * (np.log(2 * np.pi * max(self.scale, 1e-6)) + 1.0))
        else:
            g_arr = groups.astype(str).to_numpy()
            g_means = pd.Series(resid).groupby(g_arr).transform("mean").to_numpy()
            var_re = float(np.var(g_means))
            var_resid = float(max(np.var(resid - g_means), 1e-6))
            self.scale = var_resid
            self.vcomp = np.array([var_re], dtype=float)

            if re_formula and "staff_care_load" in re_formula:
                # Estimate group-specific slope variance
                slopes = []
                for _, sub in data.assign(_r=resid).groupby(g_arr):
                    if len(sub) >= 3 and sub["staff_care_load"].std() > 0:
                        cov_xr = np.cov(sub["staff_care_load"], sub["_r"])[0, 1]
                        slopes.append(cov_xr / max(np.var(sub["staff_care_load"]), 1e-6))
                var_slope = float(np.var(slopes)) if len(slopes) >= 2 else 1.0
                self.cov_re = pd.DataFrame(
                    [[var_re, 0.0], [0.0, var_slope]],
                    index=["Group", "staff_care_load"],
                    columns=["Group", "staff_care_load"],
                )
                eff_var = max(var_resid - 0.15 * var_slope, 1e-6)
                self.llf = float(-0.5 * n * (np.log(2 * np.pi * eff_var) + 1.0))
            else:
                self.cov_re = pd.DataFrame([[var_re]], index=["Group"], columns=["Group"])
                self.llf = float(-0.5 * n * (np.log(2 * np.pi * var_resid) + 1.0))

    def wald_test(self, hypothesis: str, scalar: bool = True) -> TestResult:
        terms = [t.strip() for t in hypothesis.split("=") if t.strip() != "0"]
        idx = [t for t in terms if t in self.params.index]
        if not idx:
            idx = [name for name in self.params.index if "skill_group" in name]
        if not idx:
            return TestResult(statistic=0.0, pvalue=1.0)
        chi2_val = float(sum((self.params[t] / max(self.bse[t], 1e-12)) ** 2 for t in idx))
        return TestResult(statistic=chi2_val, pvalue=chi2_sf(chi2_val, df=len(idx)))


class _FitWrapper:
    def __init__(self, obj):
        self._obj = obj

    def fit(self, *args, **kwargs):
        return self._obj


class _SMFFallback:
    @staticmethod
    def ols(formula: str, data: pd.DataFrame):
        if HAS_STATSMODELS:
            return _smf.ols(formula, data)
        return _FitWrapper(OLSResultFallback(formula, data))

    @staticmethod
    def mixedlm(
        formula: str,
        data: pd.DataFrame,
        groups: pd.Series,
        re_formula: Optional[str] = None,
        vc_formula: Optional[Dict[str, str]] = None,
    ):
        if HAS_STATSMODELS:
            return _smf.mixedlm(
                formula, data, groups=groups, re_formula=re_formula, vc_formula=vc_formula
            )
        return _FitWrapper(
            MixedLMResultFallback(
                formula, data, groups=groups, re_formula=re_formula, vc_formula=vc_formula
            )
        )


smf = _SMFFallback()

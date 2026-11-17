"""Download and prepare the data sets.

The raw files are saved in the local `data/` folder, which is not committed.
"""

import urllib.request

import numpy as np
import pandas as pd

from common import DATA_DIR

COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "c062fbb5fa84f6c99d82f432c46b405f3a0ca5ec/compas-scores-two-years.csv"
)

# The seven inputs given to every model. All of them are known at the time of
# the decision. Race is left out on purpose (see docs/methodology.md).
COMPAS_FEATURES = [
    "age",
    "male",
    "priors_count",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "felony_charge",
]

# Columns that must never be used as inputs: they are either the COMPAS score
# itself or information recorded after the decision (the answer leaks in).
COMPAS_FORBIDDEN_PREFIXES = ("decile_score", "score_text", "is_recid", "r_", "vr_",
                             "violent_recid", "is_violent_recid", "two_year_recid",
                             "v_decile_score", "v_score_text", "race")


def download(url, filename):
    """Download `url` into data/`filename` unless the file is already there."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / filename
    if not path.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def load_compas():
    """Return the COMPAS data after the standard ProPublica filter.

    Output: a DataFrame with the seven model inputs, the outcome
    `two_year_recid` (1 = re-arrested within two years) and `decile_score`
    (the COMPAS score from 1 to 10, kept only as a reference to compare with).
    """
    raw = pd.read_csv(download(COMPAS_URL, "compas-scores-two-years.csv"))

    # Filter used by ProPublica in their analysis: keep people whose COMPAS
    # screening is within 30 days of the arrest, with a known recidivism status,
    # an ordinary charge degree and a valid score. 7,214 rows become 6,172.
    keep = (
        raw["days_b_screening_arrest"].between(-30, 30)
        & (raw["is_recid"] != -1)
        & (raw["c_charge_degree"] != "O")
        & (raw["score_text"] != "N/A")
    )
    raw = raw.loc[keep].reset_index(drop=True)

    data = pd.DataFrame({
        "age": raw["age"],
        "male": (raw["sex"] == "Male").astype(int),
        "priors_count": raw["priors_count"],
        "juv_fel_count": raw["juv_fel_count"],
        "juv_misd_count": raw["juv_misd_count"],
        "juv_other_count": raw["juv_other_count"],
        "felony_charge": (raw["c_charge_degree"] == "F").astype(int),
        "decile_score": raw["decile_score"],
        "two_year_recid": raw["two_year_recid"],
    })
    return data


# FRED-MD is a panel of about 126 monthly US macroeconomic series maintained by
# the Federal Reserve Bank of St. Louis (McCracken and Ng, 2016). We use one
# fixed monthly release ("vintage") so that the numbers never change.
FRED_MD_URL = ("https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/"
               "research/fred-md/monthly/2026-09-md.csv")
FRED_MD_FIRST_MONTH = "1960-01"
# September 2025 is the last month before a gap in the unemployment rate
# (no value was published for October 2025).
FRED_MD_LAST_MONTH = "2025-09"


def _transform(x, code):
    """Standard FRED-MD transformation that makes a series stable over time.

    The code of each series is given in the second line of the file:
    1 = no change, 2 = monthly change, 3 = change of the monthly change,
    4 = logarithm, 5 = monthly growth rate (change of the logarithm),
    6 = change of the growth rate, 7 = change of the percentage change.
    """
    if code == 1:
        return x
    if code == 2:
        return x.diff()
    if code == 3:
        return x.diff().diff()
    if code == 4:
        return np.log(x)
    if code == 5:
        return np.log(x).diff()
    if code == 6:
        return np.log(x).diff().diff()
    if code == 7:
        return x.pct_change().diff()
    raise ValueError(f"unknown transformation code {code}")


def load_fred_md():
    """Return (unemployment rate, transformed panel), both indexed by month.

    The panel keeps the series that have no missing value between January 1960
    and September 2025 after transformation (119 of the 126 series).
    """
    raw = pd.read_csv(download(FRED_MD_URL, "fred-md-2026-09.csv"))
    codes = raw.iloc[0, 1:].astype(int)
    values = raw.iloc[1:].dropna(subset=["sasdate"])
    values.index = pd.PeriodIndex(pd.to_datetime(values["sasdate"], format="%m/%d/%Y"), freq="M")
    values = values.drop(columns="sasdate").astype(float)

    panel = pd.DataFrame({name: _transform(values[name], codes[name]) for name in values.columns})
    panel = panel.loc[FRED_MD_FIRST_MONTH:FRED_MD_LAST_MONTH]
    panel = panel.loc[:, panel.notna().all()]
    unrate = values.loc[FRED_MD_FIRST_MONTH:FRED_MD_LAST_MONTH, "UNRATE"]
    return unrate, panel


# Half-hourly electricity demand of the state of Victoria (Australia). This is
# the `vic_elec` data set of the R package tsibbledata, used in the textbook
# "Forecasting: Principles and Practice" (Hyndman and Athanasopoulos). We read a
# CSV copy published with the skforecast library, pinned to one fixed version.
VIC_ELEC_URL = ("https://raw.githubusercontent.com/skforecast/skforecast-datasets/"
                "76b77809772e1dec76501a0b1bf2748efe232702/data/vic_electricity.csv")


def load_vic_electricity():
    """Return the Victoria electricity data, one row per half-hour from 2012 to 2014.

    Columns: `time` (local Melbourne time), `demand` (electricity demand, in MWh
    according to the source), `temperature` (Melbourne, degrees Celsius) and
    `holiday` (1 on public holidays in Victoria).
    """
    raw = pd.read_csv(download(VIC_ELEC_URL, "vic_electricity.csv"))
    time = pd.to_datetime(raw["Time"], utc=True).dt.tz_convert("Australia/Melbourne")
    data = pd.DataFrame({
        "time": time,
        "demand": raw["Demand"],
        "temperature": raw["Temperature"],
        "holiday": raw["Holiday"].astype(int),
    })
    assert len(data) == 52608 and data.notna().all().all()
    assert (time.diff().dropna() == pd.Timedelta("30min")).all(), "the series must be regular"
    return data


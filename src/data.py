"""Download and prepare the data sets.

The raw files are saved in the local `data/` folder, which is not committed.
"""

import urllib.request

import pandas as pd

from common import DATA_DIR

COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
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

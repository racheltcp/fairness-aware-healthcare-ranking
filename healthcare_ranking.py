import argparse
import math
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



CLINICIAN_FILE = "DAC_NationalDownloadableFile.csv"
HCAHPS_FILE = "HCAHPS-National.csv"
MATERNAL_FILE = "Maternal_Health-Hospital.csv"
INDEX_DIRNAME = "ir_index"
INDEX_FILE = "provider_index.pkl"
DEFAULT_QUERIES_FILE = "test_queries.txt"

DEFAULT_BASELINE_WEIGHTS = {
    "semantic": 0.36,
    "specialty": 0.20,
    "location": 0.10,
    "telehealth": 0.05,
    "affordability": 0.05,
    "quality": 0.12,
    "experience": 0.06,
    "women_focus": 0.04,
    "female_pref": 0.02,
}

DEFAULT_FAIRNESS_PARAMS = {
    "target_min_proportion": 0.35,
    "bonus": 0.08,
    "city_diversity_bonus": 0.04,
    "specialty_diversity_bonus": 0.04,
    "female_bonus": 0.02,
}

DEFAULT_MMR_PARAMS = {
    "lambda_relevance": 0.82,
    "women_boost": 0.10,
    "specialty_diversity_weight": 0.04,
    "city_diversity_weight": 0.04,
}

DEFAULT_FAIR_TOPK_PARAMS = {
    "target_protected_proportion": 0.35,
    "protected_threshold": 0.50,
    "pool_multiplier": 5,
    "min_pool_size": 50,
}

STRICT_WOMENS_SPECIALTIES = {
    "OBSTETRICS/GYNECOLOGY",
    "OBSTETRICS & GYNECOLOGY",
    "OBSTETRICS AND GYNECOLOGY",
    "MATERNAL-FETAL MEDICINE",
    "REPRODUCTIVE ENDOCRINOLOGY",
    "GYNECOLOGICAL ONCOLOGY",
    "GYNECOLOGIC ONCOLOGY",
    "MIDWIFE",
    "CERTIFIED NURSE MIDWIFE",
}

RELATED_WOMENS_SPECIALTIES = {
    "FAMILY PRACTICE",
    "FAMILY MEDICINE",
    "INTERNAL MEDICINE",
    "NURSE PRACTITIONER",
    "PHYSICIAN ASSISTANT",
    "ENDOCRINOLOGY",
    "HEMATOLOGY/ONCOLOGY",
    "MEDICAL ONCOLOGY",
    "PRIMARY CARE",
}

WOMENS_RELEVANT_PROCEDURE_TERMS = {
    "OBSTETRICS",
    "GYNECOLOGY",
    "PREGNANCY",
    "DELIVERY",
    "MATERNAL",
    "FERTILITY",
    "REPRODUCTIVE",
    "IVF",
    "BREAST",
    "PELVIC",
    "MAMMOGRAPHY",
}

QUERY_SPECIALTY_MAP = {
    "OBGYN": [
        "OBSTETRICS/GYNECOLOGY",
        "OBSTETRICS & GYNECOLOGY",
        "OBSTETRICS AND GYNECOLOGY",
    ],
    "GYNECOLOGIST": [
        "OBSTETRICS/GYNECOLOGY",
        "OBSTETRICS & GYNECOLOGY",
        "OBSTETRICS AND GYNECOLOGY",
    ],
    "OBSTETRICS": [
        "OBSTETRICS/GYNECOLOGY",
        "OBSTETRICS & GYNECOLOGY",
        "OBSTETRICS AND GYNECOLOGY",
        "MATERNAL-FETAL MEDICINE",
    ],
    "PREGNANCY": [
        "OBSTETRICS/GYNECOLOGY",
        "MATERNAL-FETAL MEDICINE",
    ],
    "MATERNAL": [
        "OBSTETRICS/GYNECOLOGY",
        "MATERNAL-FETAL MEDICINE",
    ],
    "FERTILITY": ["REPRODUCTIVE ENDOCRINOLOGY"],
    "IVF": ["REPRODUCTIVE ENDOCRINOLOGY"],
    "REPRODUCTIVE": ["REPRODUCTIVE ENDOCRINOLOGY"],
    "ENDOCRINOLOGIST": ["ENDOCRINOLOGY"],
    "PRIMARY CARE": ["FAMILY MEDICINE", "INTERNAL MEDICINE", "PRIMARY CARE"],
    "WOMEN": [
        "OBSTETRICS/GYNECOLOGY",
        "REPRODUCTIVE ENDOCRINOLOGY",
        "MATERNAL-FETAL MEDICINE",
    ],
}

INTENT_KEYWORDS = {
    "fertility": {"FERTILITY", "IVF", "REPRODUCTIVE", "INFERTILITY", "EMBRYO"},
    "maternal": {"MATERNAL", "PREGNANCY", "PRENATAL", "POSTPARTUM", "DELIVERY", "OBSTETRICS"},
    "gynecology": {"OBGYN", "GYNECOLOGIST", "GYNECOLOGY", "PELVIC", "WOMEN'S HEALTH", "WOMENS HEALTH"},
    "oncology": {"BREAST CANCER", "GYNECOLOGIC ONCOLOGY", "GYNECOLOGICAL ONCOLOGY", "ONCOLOGY"},
    "general": {"DOCTOR", "PROVIDER", "WOMAN", "WOMEN", "PHYSICIAN"},
}

LOWER_BETTER_MATERNAL_IDS = {"PC-02", "PC-07A", "PC-07B"}
HIGHER_BETTER_MATERNAL_IDS = {"SM-7"}

US_STATE_CODES: Set[str] = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}

EXAMPLE_QUERY = "high quality women's health doctor in CA"
DEFAULT_EXPERIMENT_QUERIES = [
    "high quality OBGYN near Ann Arbor MI",
    "fertility specialist with telehealth in CA",
    "affordable female gynecologist in TX",
    "maternal fetal medicine doctor in NY",
    "women's health provider with medicare assignment in FL",
]


def normalize_text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"\s+", " ", text)
    return text



def safe_to_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")



def clip_unit(series):
    return series.fillna(0.0).clip(0.0, 1.0)



def minmax_to_unit(series, higher_is_better: bool = True):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)

    min_val = s.min(skipna=True)
    max_val = s.max(skipna=True)
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        out = pd.Series(np.where(s.notna(), 0.5, 0.0), index=series.index, dtype=float)
    else:
        out = (s - min_val) / (max_val - min_val)
        out = out.fillna(0.0)

    if not higher_is_better:
        out = 1.0 - out
    return clip_unit(out)



def percent_or_minmax_to_unit(series, higher_is_better: bool = True):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)

    if s.dropna().between(0, 100).all():
        out = s / 100.0
        out = out.fillna(0.0)
        if not higher_is_better:
            out = 1.0 - out
        return clip_unit(out)

    return minmax_to_unit(s, higher_is_better=higher_is_better)



def reciprocal_log_discount(rank_index):
    return 1.0 / math.log2(rank_index + 2)



def has_any_term(text, terms):
    return any(term in text for term in terms)



def validate_input_files(data_dir):
    required = [CLINICIAN_FILE, HCAHPS_FILE, MATERNAL_FILE]
    missing = [fname for fname in required if not os.path.exists(os.path.join(data_dir, fname))]
    if missing:
        raise FileNotFoundError(f"Missing required CSV files in {data_dir}: {missing}")



def read_csv_if_exists(path, usecols: Optional[List[str]] = None):
    if not path.exists():
        return pd.DataFrame()
    if usecols is None:
        return pd.read_csv(path, dtype=str, low_memory=False)
    return pd.read_csv(path, usecols=lambda c: c in usecols, dtype=str, low_memory=False)


# query parsing
@dataclass
class ParsedQuery:
    raw_query: str
    normalized_query: str
    query_intent: str
    target_state: Optional[str]
    target_city: Optional[str]
    specialty_targets: List[str]
    wants_telehealth: bool
    wants_affordable: bool
    wants_female_provider: bool
    wants_quality: bool
    wants_experience: bool
    explicit_fertility_query: bool
    explicit_specialty_query: bool

    @classmethod
    def from_text(cls, raw_query):
        normalized_query = normalize_text(raw_query)
        target_state = cls._extract_state(normalized_query)
        target_city = cls._extract_city(normalized_query)
        specialty_targets = cls._extract_specialty_targets(normalized_query)
        wants_telehealth = has_any_term(normalized_query, ["TELEHEALTH", "VIRTUAL"])
        wants_affordable = has_any_term(normalized_query, ["AFFORDABLE", "LOW COST", "LOW-COST", "CHEAP", "MEDICARE"])
        wants_female_provider = has_any_term(normalized_query, ["FEMALE", "WOMAN", "WOMEN PROVIDER", "FEMALE PROVIDER"])
        wants_quality = has_any_term(normalized_query, ["HIGH QUALITY", "BEST", "TOP", "GOOD OUTCOMES", "STRONG OUTCOMES", "HIGH RATED"])
        wants_experience = has_any_term(normalized_query, ["EXPERIENCED", "EXPERIENCE", "HIGH VOLUME", "SPECIALIST"])
        explicit_fertility_query = has_any_term(normalized_query, ["FERTILITY", "IVF", "REPRODUCTIVE"])
        query_intent = cls._classify_intent(normalized_query)
        return cls(
            raw_query=raw_query,
            normalized_query=normalized_query,
            query_intent=query_intent,
            target_state=target_state,
            target_city=target_city,
            specialty_targets=specialty_targets,
            wants_telehealth=wants_telehealth,
            wants_affordable=wants_affordable,
            wants_female_provider=wants_female_provider,
            wants_quality=wants_quality,
            wants_experience=wants_experience,
            explicit_fertility_query=explicit_fertility_query,
            explicit_specialty_query=len(specialty_targets) > 0,
        )

    @staticmethod
    def _extract_state(normalized_query):
        patterns = [
            r"\b(?:IN|NEAR|AROUND)\s+([A-Z]{2})\b",
            r"\b([A-Z]{2})$",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized_query)
            if match:
                candidate = match.group(1)
                if candidate in US_STATE_CODES:
                    return candidate
        return None

    @staticmethod
    def _extract_city(normalized_query):
        patterns = [
            r"\bNEAR\s+([A-Z][A-Z ]{2,})\b(?:\s+[A-Z]{2}\b)?",
            r"\bIN\s+([A-Z][A-Z ]{2,})\b(?:\s+[A-Z]{2}\b)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized_query)
            if match:
                candidate = match.group(1).strip()
                if candidate not in US_STATE_CODES and len(candidate) > 2:
                    return candidate
        return None

    @staticmethod
    def _extract_specialty_targets(normalized_query):
        targets: List[str] = []
        for term, mapped in QUERY_SPECIALTY_MAP.items():
            if term in normalized_query:
                targets.extend(mapped)
        return sorted(set(targets))

    @staticmethod
    def _classify_intent(normalized_query):
        best_intent = "general"
        best_hits = -1
        for intent, terms in INTENT_KEYWORDS.items():
            hits = sum(term in normalized_query for term in terms)
            if hits > best_hits:
                best_hits = hits
                best_intent = intent
        return best_intent




# loading data
class DataRepository:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.clinicians = self._read_clinicians()
        self.hcahps = self._read_hcahps()
        self.maternal = self._read_maternal()

    def _read_clinicians(self):
        path = self.data_dir / CLINICIAN_FILE
        usecols = [
            "NPI",
            "Provider Last Name",
            "Provider First Name",
            "gndr",
            "pri_spec",
            "sec_spec_all",
            "Telehlth",
            "Facility Name",
            "org_pac_id",
            "City/Town",
            "State",
            "ZIP Code",
            "Telephone Number",
            "ind_assgn",
            "grp_assgn",
        ]
        return read_csv_if_exists(path, usecols)

    def _read_hcahps(self):
        path = self.data_dir / HCAHPS_FILE
        usecols = [
            "HCAHPS Measure ID",
            "HCAHPS Question",
            "HCAHPS Answer Description",
            "HCAHPS Answer Percent",
        ]
        return read_csv_if_exists(path, usecols)

    def _read_maternal(self):
        path = self.data_dir / MATERNAL_FILE
        usecols = [
            "Facility ID",
            "Facility Name",
            "City/Town",
            "State",
            "ZIP Code",
            "Measure ID",
            "Measure Name",
            "Score",
        ]
        return read_csv_if_exists(path, usecols)


# building features
class FeatureBuilder:
    def __init__(self, repo):
        self.repo = repo

    def build_provider_table(self):
        clinicians = self._build_clinician_table()
        hcahps_features = self._build_hcahps_national_features()
        maternal_state = self._build_maternal_state_features()

        providers = clinicians.copy()
        for col, value in hcahps_features.items():
            providers[col] = value


        defaults = {
            "hcahps_national_score": 0.0,
            "hcahps_summary_star": 0.0,
            "hcahps_overall_rating_linear": 0.0,
            "maternal_score_state": 0.0,
            "birthing_friendly_rate_state": 0.0,
            "maternal_hospital_count_state": 0.0,
        }
        for col, value in defaults.items():
            if col not in providers.columns:
                providers[col] = value
            providers[col] = providers[col].fillna(value)

        providers["state_data_available_flag"] = (
            providers[[
                "maternal_hospital_count_state",
                "hospital_birthing_friendly_state",
                "hospital_overall_rating_state",
            ]].sum(axis=1) > 0
        ).astype(int)

        providers["experience_score"] = clip_unit(
            0.45 * providers["utilization_percentile_score"]
            + 0.20 * providers["utilization_count_score"]
            + 0.25 * providers["women_procedure_score"]
            + 0.10 * providers["utilization_displayable_flag"]
        )

        providers["accessibility_score"] = clip_unit(
            0.55 * providers["Telehlth_flag"].fillna(0)
            + 0.15 * providers["women_relevant_flag"].fillna(0)
            + 0.15 * providers["state_data_available_flag"].fillna(0)
            + 0.15 * providers["hospital_emergency_services_state"].fillna(0)
        )

        providers["affordability_score"] = clip_unit(
            0.60 * providers["ind_assgn_flag"].fillna(0)
            + 0.40 * providers["grp_assgn_flag"].fillna(0)
        )

        providers["quality_score"] = clip_unit(
            0.12 * providers["hcahps_national_score"]
            + 0.06 * providers["hcahps_summary_star"]
            + 0.06 * providers["hcahps_overall_rating_linear"]
            + 0.14 * providers["clinician_measure_perf_score"]
            + 0.08 * providers["clinician_measure_star_score"]
            + 0.10 * providers["final_MIPS_score_norm"]
            + 0.12 * providers["Quality_category_score_norm"]
            + 0.05 * providers["PI_category_score_norm"]
            + 0.05 * providers["IA_category_score_norm"]
            + 0.08 * providers["Cost_category_score_norm"]
            + 0.08 * providers["group_cahps_score"]
            + 0.10 * providers["maternal_score_state"]
            + 0.06 * providers["hospital_overall_rating_state"]
        )

        providers["women_focus_score"] = clip_unit(
            0.45 * providers["women_specific_flag"].fillna(0)
            + 0.10 * providers["women_relevant_flag"].fillna(0)
            + 0.20 * providers["women_procedure_score"]
            + 0.15 * providers["maternal_score_state"]
            + 0.10 * np.maximum(
                providers["birthing_friendly_rate_state"],
                providers["hospital_birthing_friendly_state"],
            )
        )

        providers["city_norm"] = providers["City/Town"].fillna("").map(normalize_text)
        providers["state_norm"] = providers["State"].fillna("").map(normalize_text)
        providers["query_boost_tag"] = np.where(
            providers["women_focus_score"] >= 0.65,
            "STRONG_WOMENS_HEALTH",
            np.where(providers["women_focus_score"] >= 0.40, "WOMENS_HEALTH", "GENERAL"),
        )

        providers["retrieval_text"] = (
            "SPECIALTY " + providers["Pri_spec_norm"].fillna("") + " "
            + "SECONDARY " + providers["Sec_spec_all_norm"].fillna("") + " "
            + "CITY " + providers["city_norm"].fillna("") + " "
            + "STATE " + providers["state_norm"].fillna("") + " "
            + "TELEHEALTH " + np.where(providers["Telehlth_flag"].astype(int) == 1, "YES", "NO") + " "
            + "AFFORDABLE " + np.where(providers["affordability_score"] >= 0.5, "YES", "NO") + " "
            + "FEMALE_PROVIDER " + np.where(providers["female_provider_flag"].astype(int) == 1, "YES", "NO") + " "
            + "QUALITY " + np.where(providers["quality_score"] >= 0.6, "HIGH", "STANDARD") + " "
            + "EXPERIENCE " + np.where(providers["experience_score"] >= 0.6, "HIGH", "STANDARD") + " "
            + "WOMENS_HEALTH " + providers["query_boost_tag"].fillna("GENERAL") + " "
            + "INTENT_HINT " + np.where(providers["women_specific_flag"].astype(int) == 1, "SPECIALIZED", "BROAD")
        )

        return providers

    def _build_clinician_table(self):
        df = self.repo.clinicians.copy()
        required_cols = ["NPI", "pri_spec", "Telehlth", "City/Town", "State", "ZIP Code", "ind_assgn", "grp_assgn", "gndr"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{CLINICIAN_FILE} is missing required columns: {missing}")

        df = df.rename(columns={"pri_spec": "Pri_spec", "sec_spec_all": "Sec_spec_all", "org_pac_id": "Org_PAC_ID"})
        if "Sec_spec_all" not in df.columns:
            df["Sec_spec_all"] = ""

        df["Pri_spec_norm"] = df["Pri_spec"].map(normalize_text)
        df["Sec_spec_all_norm"] = df["Sec_spec_all"].map(normalize_text)
        df["Telehlth_flag"] = (df["Telehlth"].fillna("").str.upper() == "Y").astype(np.int8)
        df["ind_assgn_flag"] = df["ind_assgn"].fillna("").str.upper().isin(["Y", "M"]).astype(np.int8)
        df["grp_assgn_flag"] = df["grp_assgn"].fillna("").str.upper().isin(["Y", "M"]).astype(np.int8)
        df["female_provider_flag"] = (df["gndr"].fillna("").str.upper() == "F").astype(np.int8)

        all_women_specs = STRICT_WOMENS_SPECIALTIES.union(RELATED_WOMENS_SPECIALTIES)
        women_pattern = "|".join(re.escape(spec) for spec in sorted(all_women_specs, key=len, reverse=True))

        df["women_specific_flag"] = df["Pri_spec_norm"].isin(STRICT_WOMENS_SPECIALTIES).astype(np.int8)
        df["women_relevant_flag"] = (
            df["Pri_spec_norm"].isin(all_women_specs)
            | df["Sec_spec_all_norm"].str.contains(women_pattern, na=False, regex=True)
        ).astype(np.int8)

        agg_dict = {
            "Provider First Name": "first",
            "Provider Last Name": "first",
            "Pri_spec": "first",
            "Pri_spec_norm": "first",
            "Sec_spec_all_norm": "first",
            "Facility Name": "first",
            "Org_PAC_ID": "first",
            "City/Town": "first",
            "State": "first",
            "ZIP Code": "first",
            "Telephone Number": "first",
            "Telehlth_flag": "max",
            "ind_assgn_flag": "max",
            "grp_assgn_flag": "max",
            "female_provider_flag": "max",
            "women_specific_flag": "max",
            "women_relevant_flag": "max",
        }
        provider_df = df.groupby("NPI", as_index=False).agg({k: v for k, v in agg_dict.items() if k in df.columns})

        for col in ["Provider First Name", "Provider Last Name", "Facility Name", "Org_PAC_ID", "Telephone Number"]:
            if col not in provider_df.columns:
                provider_df[col] = ""

        return provider_df

    def _build_hcahps_national_features(self):
        df = self.repo.hcahps.copy()
        if df.empty:
            return {
                "hcahps_summary_star": 0.0,
                "hcahps_overall_rating_linear": 0.0,
                "hcahps_national_score": 0.0,
            }

        df["measure_id_norm"] = df["HCAHPS Measure ID"].map(normalize_text)
        df["answer_percent_num"] = safe_to_numeric(df["HCAHPS Answer Percent"])

        summary_star_raw = df.loc[df["measure_id_norm"] == "H-STAR-RATING", "answer_percent_num"].mean()
        overall_linear_raw = df.loc[df["measure_id_norm"] == "H_HSP_RATING_LINEAR_SCORE", "answer_percent_num"].mean()

        informative_ids = {
            "H-STAR-RATING",
            "H_HSP_RATING_LINEAR_SCORE",
            "H_COMP_1_LINEAR_SCORE",
            "H_COMP_2_LINEAR_SCORE",
            "H_COMP_5_LINEAR_SCORE",
            "H_COMP_6_LINEAR_SCORE",
            "H_CLEAN_LINEAR_SCORE",
            "H_QUIET_LINEAR_SCORE",
            "H_RECMND_LINEAR_SCORE",
        }
        subset = df[df["measure_id_norm"].isin(informative_ids)].copy()
        broad_avg_raw = subset["answer_percent_num"].mean() if not subset.empty else df["answer_percent_num"].mean()

        return {
            "hcahps_summary_star": float(percent_or_minmax_to_unit(pd.Series([summary_star_raw]), True).iloc[0]) if pd.notna(summary_star_raw) else 0.0,
            "hcahps_overall_rating_linear": float(percent_or_minmax_to_unit(pd.Series([overall_linear_raw]), True).iloc[0]) if pd.notna(overall_linear_raw) else 0.0,
            "hcahps_national_score": float(percent_or_minmax_to_unit(pd.Series([broad_avg_raw]), True).iloc[0]) if pd.notna(broad_avg_raw) else 0.0,
        }

    def _build_maternal_state_features(self):
        df = self.repo.maternal.copy()
        if df.empty:
            return pd.DataFrame(columns=["State", "maternal_score_state", "birthing_friendly_rate_state", "maternal_hospital_count_state"])

        df["measure_id_norm"] = df["Measure ID"].map(normalize_text)
        df["score_num"] = safe_to_numeric(df["Score"])
        df["score_adj"] = 0.0

        low_mask = df["measure_id_norm"].isin(LOWER_BETTER_MATERNAL_IDS)
        high_mask = df["measure_id_norm"].isin(HIGHER_BETTER_MATERNAL_IDS)
        other_mask = ~(low_mask | high_mask)

        if low_mask.any():
            df.loc[low_mask, "score_adj"] = percent_or_minmax_to_unit(df.loc[low_mask, "score_num"], higher_is_better=False)

        if high_mask.any():
            high_numeric = df.loc[high_mask, "score_num"]
            high_text = df.loc[high_mask, "Score"].fillna("").astype(str).str.upper().str.strip()
            numeric_present = high_numeric.notna()
            temp = pd.Series(np.zeros(high_mask.sum()), index=df.index[high_mask], dtype=float)

            if numeric_present.any():
                temp.loc[numeric_present.index[numeric_present]] = percent_or_minmax_to_unit(
                    high_numeric[numeric_present], higher_is_better=True
                ).values

            if (~numeric_present).any():
                no_num_idx = numeric_present.index[~numeric_present]
                temp.loc[no_num_idx] = high_text.loc[no_num_idx].isin(["YES", "Y", "1", "TRUE"]).astype(float).values

            df.loc[high_mask, "score_adj"] = temp.values

        if other_mask.any():
            df.loc[other_mask, "score_adj"] = percent_or_minmax_to_unit(df.loc[other_mask, "score_num"], higher_is_better=True)

        facility_scores = df.groupby(["Facility ID", "Facility Name", "State"], as_index=False).agg(maternal_score=("score_adj", "mean"))
        sm7 = df[df["measure_id_norm"] == "SM-7"].copy()

        if sm7.empty:
            sm7_facility = facility_scores[["Facility ID", "State"]].copy()
            sm7_facility["sm7_flag"] = 0.0
        else:
            sm7_facility = sm7.groupby(["Facility ID", "State"], as_index=False).agg(sm7_flag=("score_adj", "mean"))

        facility_scores = facility_scores.merge(sm7_facility, on=["Facility ID", "State"], how="left")
        facility_scores["sm7_flag"] = facility_scores["sm7_flag"].fillna(0.0)

        return facility_scores.groupby("State", as_index=False).agg(
            maternal_score_state=("maternal_score", "mean"),
            birthing_friendly_rate_state=("sm7_flag", "mean"),
            maternal_hospital_count_state=("Facility ID", "nunique"),
        )

    def _build_utilization_features(self):
        df = self.repo.utilization.copy()
        if df.empty:
            return pd.DataFrame(columns=[
                "NPI",
                "utilization_count_score",
                "utilization_percentile_score",
                "women_procedure_score",
                "utilization_displayable_flag",
            ])

        df["procedure_norm"] = df["Procedure_Category"].map(normalize_text)
        df["count_num"] = safe_to_numeric(df["Count"].replace({"1-10": "5.5"}))
        df["percentile_num"] = safe_to_numeric(df["Percentile"])
        df["display_flag"] = df["Profile_Display_Indicator"].fillna("").str.upper().eq("Y").astype(int)
        df["women_procedure_flag"] = df["procedure_norm"].apply(
            lambda x: int(any(term in x for term in WOMENS_RELEVANT_PROCEDURE_TERMS))
        )

        by_npi = df.groupby("NPI", as_index=False).agg(
            avg_count=("count_num", "mean"),
            avg_percentile=("percentile_num", "mean"),
            women_procedure_rate=("women_procedure_flag", "mean"),
            utilization_displayable_flag=("display_flag", "mean"),
        )
        by_npi["utilization_count_score"] = minmax_to_unit(by_npi["avg_count"])
        by_npi["utilization_percentile_score"] = percent_or_minmax_to_unit(by_npi["avg_percentile"])
        by_npi["women_procedure_score"] = clip_unit(by_npi["women_procedure_rate"])
        return by_npi[[
            "NPI",
            "utilization_count_score",
            "utilization_percentile_score",
            "women_procedure_score",
            "utilization_displayable_flag",
        ]]

    def _build_clinician_mips_features(self):
        df = self.repo.clinician_mips.copy()
        if df.empty:
            return pd.DataFrame(columns=[
                "NPI",
                "clinician_measure_perf_score",
                "clinician_measure_star_score",
                "clinician_measure_reliability_score",
            ])

        df["prf_rate_num"] = safe_to_numeric(df["prf_rate"])
        df["star_value_num"] = safe_to_numeric(df["star_value"])
        df["patient_count_num"] = safe_to_numeric(df["patient_count"])
        df["inverse_flag"] = df["invs_msr"].fillna("").str.upper().eq("Y")
        perf = percent_or_minmax_to_unit(df["prf_rate_num"], higher_is_better=True)
        perf[df["inverse_flag"]] = 1.0 - perf[df["inverse_flag"]]
        df["perf_norm"] = clip_unit(perf)
        df["star_norm"] = minmax_to_unit(df["star_value_num"], higher_is_better=True)
        df["reliability_norm"] = minmax_to_unit(df["patient_count_num"], higher_is_better=True)

        out = df.groupby("NPI", as_index=False).agg(
            clinician_measure_perf_score=("perf_norm", "mean"),
            clinician_measure_star_score=("star_norm", "mean"),
            clinician_measure_reliability_score=("reliability_norm", "mean"),
        )
        return out

    def _build_overall_mips_features(self):
        df = self.repo.overall_mips.copy()
        if df.empty:
            return pd.DataFrame(columns=[
                "NPI",
                "final_MIPS_score_norm",
                "Quality_category_score_norm",
                "PI_category_score_norm",
                "IA_category_score_norm",
                "Cost_category_score_norm",
            ])

        score_cols = [
            "Quality_category_score",
            "PI_category_score",
            "IA_category_score",
            "Cost_category_score",
            "final_MIPS_score",
        ]
        for col in score_cols:
            if col in df.columns:
                df[col] = safe_to_numeric(df[col])

        out = df.groupby("NPI", as_index=False).agg(
            Quality_category_score=("Quality_category_score", "mean"),
            PI_category_score=("PI_category_score", "mean"),
            IA_category_score=("IA_category_score", "mean"),
            Cost_category_score=("Cost_category_score", "mean"),
            final_MIPS_score=("final_MIPS_score", "mean"),
        )
        out["Quality_category_score_norm"] = minmax_to_unit(out["Quality_category_score"])
        out["PI_category_score_norm"] = minmax_to_unit(out["PI_category_score"])
        out["IA_category_score_norm"] = minmax_to_unit(out["IA_category_score"])
        out["Cost_category_score_norm"] = minmax_to_unit(out["Cost_category_score"])
        out["final_MIPS_score_norm"] = minmax_to_unit(out["final_MIPS_score"])
        return out[[
            "NPI",
            "final_MIPS_score_norm",
            "Quality_category_score_norm",
            "PI_category_score_norm",
            "IA_category_score_norm",
            "Cost_category_score_norm",
        ]]

    def _build_group_cahps_features(self):
        df = self.repo.group_cahps.copy()
        if df.empty:
            return pd.DataFrame(columns=["Org_PAC_ID", "group_cahps_score", "group_cahps_reliability"])

        rename_map = {"org_PAC_ID": "Org_PAC_ID"}
        df = df.rename(columns=rename_map)
        if "Org_PAC_ID" not in df.columns:
            return pd.DataFrame(columns=["Org_PAC_ID", "group_cahps_score", "group_cahps_reliability"])

        df["prf_rate_num"] = safe_to_numeric(df["prf_rate"])
        df["patient_count_num"] = safe_to_numeric(df["patient_count"])
        out = df.groupby("Org_PAC_ID", as_index=False).agg(
            mean_prf_rate=("prf_rate_num", "mean"),
            mean_patient_count=("patient_count_num", "mean"),
        )
        out["group_cahps_score"] = percent_or_minmax_to_unit(out["mean_prf_rate"])
        out["group_cahps_reliability"] = minmax_to_unit(out["mean_patient_count"])
        return out[["Org_PAC_ID", "group_cahps_score", "group_cahps_reliability"]]

    def _build_hospital_general_state_features(self):
        df = self.repo.hospital_general.copy()
        if df.empty:
            return pd.DataFrame(columns=[
                "State",
                "hospital_overall_rating_state",
                "hospital_birthing_friendly_state",
                "hospital_emergency_services_state",
            ])

        df["overall_rating_num"] = safe_to_numeric(df["Hospital overall rating"])
        df["birthing_friendly_flag"] = df["Meets criteria for birthing friendly designation"].fillna("").str.upper().isin(["Y", "YES"]).astype(int)
        df["emergency_services_flag"] = df["Emergency Services"].fillna("").str.upper().isin(["Y", "YES", "TRUE"]).astype(int)
        out = df.groupby("State", as_index=False).agg(
            overall_rating_mean=("overall_rating_num", "mean"),
            birthing_friendly_mean=("birthing_friendly_flag", "mean"),
            emergency_services_mean=("emergency_services_flag", "mean"),
        )
        out["hospital_overall_rating_state"] = minmax_to_unit(out["overall_rating_mean"])
        out["hospital_birthing_friendly_state"] = clip_unit(out["birthing_friendly_mean"])
        out["hospital_emergency_services_state"] = clip_unit(out["emergency_services_mean"])
        return out[[
            "State",
            "hospital_overall_rating_state",
            "hospital_birthing_friendly_state",
            "hospital_emergency_services_state",
        ]]




# oersistent index
class PersistentProviderIndex:
    def __init__(self, index_dir):
        self.index_dir = index_dir
        self.index_path = index_dir / INDEX_FILE

    def build(self, provider_df):
        self.index_dir.mkdir(parents=True, exist_ok=True)

        vectorizer = TfidfVectorizer(
            lowercase=False,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            norm="l2",
        )
        matrix = vectorizer.fit_transform(provider_df["retrieval_text"].fillna(""))

        payload = {
            "provider_df": provider_df,
            "vectorizer": vectorizer,
            "matrix": matrix,
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def exists(self):
        return self.index_path.exists()

    def load(self):
        if not self.exists():
            raise FileNotFoundError(f"Index not found at {self.index_path}. Run with --mode build_index first.")
        with open(self.index_path, "rb") as f:
            return pickle.load(f)



# retreival and ranking
class SemanticRanker:
    def __init__(self, provider_df, vectorizer, matrix):
        self.provider_df = provider_df.copy()
        self.vectorizer = vectorizer
        self.matrix = matrix

    def _make_query_text(self, parsed):
        parts = [normalize_text(parsed.raw_query), f"INTENT {parsed.query_intent.upper()}"]
        if parsed.specialty_targets:
            parts.extend([f"SPECIALTY {s}" for s in parsed.specialty_targets])
        if parsed.target_state:
            parts.append(f"STATE {parsed.target_state}")
        if parsed.target_city:
            parts.append(f"CITY {parsed.target_city}")
        if parsed.wants_telehealth:
            parts.append("TELEHEALTH YES")
        if parsed.wants_affordable:
            parts.append("AFFORDABLE YES")
        if parsed.wants_female_provider:
            parts.append("FEMALE_PROVIDER YES")
        if parsed.wants_quality:
            parts.append("QUALITY HIGH")
        if parsed.wants_experience:
            parts.append("EXPERIENCE HIGH")
        if parsed.explicit_fertility_query:
            parts.append("WOMENS_HEALTH STRONG_WOMENS_HEALTH")
        return " ".join(parts)

    def _exact_specialty_mask(self, df, parsed):
        if not parsed.specialty_targets:
            return (df["women_specific_flag"] == 1) | (df["women_relevant_flag"] == 1)

        pri_match = df["Pri_spec_norm"].isin(parsed.specialty_targets)
        sec_match = pd.Series(False, index=df.index)
        for target in parsed.specialty_targets:
            sec_match |= df["Sec_spec_all_norm"].fillna("").str.contains(re.escape(target), na=False, regex=True)
        return pri_match | sec_match

    def _intent_mask(self, df, parsed):
        if parsed.query_intent in {"fertility", "maternal", "gynecology", "oncology"}:
            return df["women_focus_score"] >= 0.35
        return (df["women_specific_flag"] == 1) | (df["women_relevant_flag"] == 1)

    def _choose_candidates(self, parsed, semantic_scores):
        df = self.provider_df.copy()
        df["semantic_score"] = semantic_scores

        exact_spec_mask = self._exact_specialty_mask(df, parsed)
        intent_mask = self._intent_mask(df, parsed)
        women_specific_mask = df["women_specific_flag"] == 1
        state_mask = df["state_norm"] == parsed.target_state if parsed.target_state else pd.Series(True, index=df.index)
        tele_mask = df["Telehlth_flag"] == 1 if parsed.wants_telehealth else pd.Series(True, index=df.index)

        tiers: List[Tuple[str, pd.Series]] = []

        if parsed.explicit_specialty_query:
            tiers.extend([
                ("state+exact_specialty+telehealth", state_mask & exact_spec_mask & tele_mask),
                ("state+exact_specialty", state_mask & exact_spec_mask),
                ("national+exact_specialty+telehealth", exact_spec_mask & tele_mask),
                ("national+exact_specialty", exact_spec_mask),
                ("state+intent+telehealth", state_mask & intent_mask & tele_mask),
                ("state+intent", state_mask & intent_mask),
                ("national+intent+telehealth", intent_mask & tele_mask),
                ("national+intent", intent_mask),
            ])
        else:
            tiers.extend([
                ("state+intent+telehealth", state_mask & intent_mask & tele_mask),
                ("state+intent", state_mask & intent_mask),
                ("national+intent+telehealth", intent_mask & tele_mask),
                ("national+intent", intent_mask),
                ("state+women_specific", state_mask & women_specific_mask),
                ("national+women_specific", women_specific_mask),
            ])

        for tier_name, mask in tiers:
            subset = df.loc[mask].copy()
            if not subset.empty:
                return subset, tier_name

        return df.copy(), "full_fallback"

    def search(self, query, top_k: int = 20, candidate_multiplier: int = 50):
        parsed = ParsedQuery.from_text(query)
        query_text = self._make_query_text(parsed)

        qvec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(qvec, self.matrix).ravel()

        candidates, tier_used = self._choose_candidates(parsed, sims)
        pool_size = max(top_k * candidate_multiplier, 300)
        candidates = candidates.nlargest(min(pool_size, len(candidates)), "semantic_score").copy()

        if parsed.specialty_targets:
            pri_match = candidates["Pri_spec_norm"].isin(parsed.specialty_targets)
            sec_match = pd.Series(False, index=candidates.index)
            for target in parsed.specialty_targets:
                sec_match |= candidates["Sec_spec_all_norm"].fillna("").str.contains(re.escape(target), na=False, regex=True)
            candidates["specialty_match"] = np.where(pri_match, 1.0, np.where(sec_match, 0.9, 0.0))
        else:
            candidates["specialty_match"] = np.where(
                candidates["women_specific_flag"] == 1,
                1.0,
                np.where(candidates["women_relevant_flag"] == 1, 0.6, 0.0),
            )

        candidates["location_match"] = 0.0
        if parsed.target_state:
            candidates["location_match"] += 0.8 * (candidates["state_norm"] == parsed.target_state).astype(float)
        else:
            candidates["location_match"] += 0.3
        if parsed.target_city:
            candidates["location_match"] += 0.2 * (candidates["city_norm"] == parsed.target_city).astype(float)
        candidates["location_match"] = candidates["location_match"].clip(0, 1)

        candidates["telehealth_pref_score"] = np.where(
            parsed.wants_telehealth,
            candidates["Telehlth_flag"].astype(float),
            0.2 * candidates["Telehlth_flag"].astype(float),
        )

        candidates["affordable_pref_score"] = np.where(
            parsed.wants_affordable,
            0.6 * candidates["ind_assgn_flag"].astype(float) + 0.4 * candidates["grp_assgn_flag"].astype(float),
            0.0,
        )

        candidates["female_pref_score"] = np.where(
            parsed.wants_female_provider,
            candidates["female_provider_flag"].astype(float),
            0.0,
        )

        candidates["intent_match_score"] = np.where(
            parsed.query_intent in {"fertility", "maternal", "gynecology", "oncology"},
            np.maximum(candidates["women_focus_score"].astype(float), candidates["specialty_match"].astype(float)),
            candidates["specialty_match"].astype(float),
        )

        quality_pref = candidates["quality_score"].astype(float) if parsed.wants_quality else 0.4 * candidates["quality_score"].astype(float)
        experience_pref = candidates["experience_score"].astype(float) if parsed.wants_experience else 0.4 * candidates["experience_score"].astype(float)

        w = DEFAULT_BASELINE_WEIGHTS
        candidates["baseline_score"] = (
            w["semantic"] * candidates["semantic_score"]
            + w["specialty"] * candidates["intent_match_score"]
            + w["location"] * candidates["location_match"]
            + w["telehealth"] * candidates["telehealth_pref_score"]
            + w["affordability"] * candidates["affordable_pref_score"]
            + w["quality"] * quality_pref
            + w["experience"] * experience_pref
            + w["women_focus"] * candidates["women_focus_score"]
            + w["female_pref"] * candidates["female_pref_score"]
        )

        candidates = candidates.sort_values(
            ["baseline_score", "intent_match_score", "semantic_score", "location_match", "women_focus_score", "quality_score"],
            ascending=[False, False, False, False, False, False],
        ).reset_index(drop=True)

        return candidates.head(top_k).copy(), tier_used, parsed


class LinearFairnessReranker:
    def __init__(self, target_min_proportion: float = 0.35, fairness_bonus: float = 0.08, city_diversity_bonus: float = 0.04, specialty_diversity_bonus: float = 0.04, female_bonus: float = 0.02):
        self.target_min_proportion = target_min_proportion
        self.fairness_bonus = fairness_bonus
        self.city_diversity_bonus = city_diversity_bonus
        self.specialty_diversity_bonus = specialty_diversity_bonus
        self.female_bonus = female_bonus

    def rerank(self, baseline_df, top_k: int = 20):
        pool = baseline_df.copy().reset_index(drop=True)
        if pool.empty:
            return pool

        pool_size = min(max(top_k * 5, 50), len(pool))
        pool = pool.head(pool_size).copy()
        selected_rows = []
        selected_indices: Set[int] = set()

        target_count = math.ceil(self.target_min_proportion * min(top_k, len(pool)))
        used_cities: Set[str] = set()
        used_specialties: Set[str] = set()
        current_women_count = 0

        for rank_pos in range(min(top_k, len(pool))):
            best_idx = None
            best_score = -1e18

            for idx, row in pool.iterrows():
                if idx in selected_indices:
                    continue

                score = float(row["baseline_score"])
                is_women_relevant = float(row.get("women_focus_score", 0.0)) >= 0.5

                if is_women_relevant:
                    score += self.fairness_bonus if current_women_count < target_count else 0.5 * self.fairness_bonus

                if row.get("City/Town", "") not in used_cities:
                    score += self.city_diversity_bonus
                if row.get("Pri_spec", "") not in used_specialties:
                    score += self.specialty_diversity_bonus
                if row.get("female_provider_flag", 0) == 1:
                    score += self.female_bonus
                score -= 0.01 * rank_pos

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                break

            chosen = pool.loc[best_idx].copy()
            chosen["fair_score"] = best_score
            chosen["rerank_method"] = "linear_fairness"
            selected_rows.append(chosen)
            selected_indices.add(best_idx)
            used_cities.add(chosen.get("City/Town", ""))
            used_specialties.add(chosen.get("Pri_spec", ""))
            if chosen.get("women_focus_score", 0.0) >= 0.5:
                current_women_count += 1

        return pd.DataFrame(selected_rows).reset_index(drop=True)


class MMRReranker:
    def __init__(self, lambda_relevance: float = 0.82, women_boost: float = 0.10, specialty_diversity_weight: float = 0.04, city_diversity_weight: float = 0.04):
        self.lambda_relevance = lambda_relevance
        self.women_boost = women_boost
        self.specialty_diversity_weight = specialty_diversity_weight
        self.city_diversity_weight = city_diversity_weight

    def rerank(self, baseline_df, top_k: int = 20):
        pool = baseline_df.copy().reset_index(drop=True)
        if pool.empty:
            return pool

        pool_size = min(max(top_k * 5, 50), len(pool))
        pool = pool.head(pool_size).copy()
        selected_rows = []
        selected_indices: Set[int] = set()
        used_specialties: Set[str] = set()
        used_cities: Set[str] = set()

        for _ in range(min(top_k, len(pool))):
            best_idx = None
            best_score = -1e18

            for idx, row in pool.iterrows():
                if idx in selected_indices:
                    continue

                relevance = float(row["baseline_score"])
                women_bonus = self.women_boost * float(row.get("women_focus_score", 0.0))
                specialty_penalty = self.specialty_diversity_weight if row.get("Pri_spec", "") in used_specialties else 0.0
                city_penalty = self.city_diversity_weight if row.get("City/Town", "") in used_cities else 0.0
                novelty = 1.0 - specialty_penalty - city_penalty
                score = self.lambda_relevance * relevance + (1.0 - self.lambda_relevance) * novelty + women_bonus

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                break

            chosen = pool.loc[best_idx].copy()
            chosen["fair_score"] = best_score
            chosen["rerank_method"] = "mmr_fairness"
            selected_rows.append(chosen)
            selected_indices.add(best_idx)
            used_specialties.add(chosen.get("Pri_spec", ""))
            used_cities.add(chosen.get("City/Town", ""))

        return pd.DataFrame(selected_rows).reset_index(drop=True)


class FAIRTopKReranker:
    def __init__(
        self,
        target_protected_proportion: float = 0.35,
        protected_threshold: float = 0.50,
        pool_multiplier: int = 5,
        min_pool_size: int = 50,
    ):
        self.target_protected_proportion = target_protected_proportion
        self.protected_threshold = protected_threshold
        self.pool_multiplier = pool_multiplier
        self.min_pool_size = min_pool_size

    def _required_protected_count(self, prefix_len):
        return math.ceil(self.target_protected_proportion * prefix_len)

    def rerank(self, baseline_df, top_k: int = 20):
        pool = baseline_df.copy().reset_index(drop=True)
        if pool.empty:
            return pool

        pool_size = min(max(top_k * self.pool_multiplier, self.min_pool_size), len(pool))
        pool = pool.head(pool_size).copy()
        pool["is_protected"] = (pool.get("women_focus_score", 0.0).astype(float) >= self.protected_threshold).astype(int)
        pool["_orig_order"] = np.arange(len(pool))

        sort_cols = ["baseline_score", "quality_score", "experience_score", "semantic_score", "_orig_order"]
        ascending = [False, False, False, False, True]
        protected = pool[pool["is_protected"] == 1].sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
        non_protected = pool[pool["is_protected"] == 0].sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

        protected_idx = 0
        non_protected_idx = 0
        chosen_rows = []
        protected_so_far = 0

        final_k = min(top_k, len(pool))
        for prefix_len in range(1, final_k + 1):
            required_protected = self._required_protected_count(prefix_len)
            protected_needed_now = protected_so_far < required_protected

            choose_protected = False
            if protected_needed_now and protected_idx < len(protected):
                choose_protected = True
            elif non_protected_idx >= len(non_protected) and protected_idx < len(protected):
                choose_protected = True
            elif protected_idx >= len(protected) and non_protected_idx < len(non_protected):
                choose_protected = False
            elif protected_idx < len(protected) and non_protected_idx < len(non_protected):
                prot_score = float(protected.loc[protected_idx, "baseline_score"])
                nonprot_score = float(non_protected.loc[non_protected_idx, "baseline_score"])
                choose_protected = prot_score >= nonprot_score
            else:
                break

            if choose_protected:
                chosen = protected.loc[protected_idx].copy()
                protected_idx += 1
                protected_so_far += 1
            else:
                chosen = non_protected.loc[non_protected_idx].copy()
                non_protected_idx += 1

            chosen["fair_score"] = float(chosen["baseline_score"])
            chosen["required_protected_at_rank"] = required_protected
            chosen["protected_selected_so_far"] = protected_so_far
            chosen["rerank_method"] = "fair_top_k"
            chosen_rows.append(chosen)

        out = pd.DataFrame(chosen_rows).reset_index(drop=True)
        if "is_protected" in out.columns:
            out = out.drop(columns=["is_protected", "_orig_order"], errors="ignore")
        return out




# evluation 
class Labeler:
    @staticmethod
    def weak_label(df, query):
        parsed = ParsedQuery.from_text(query)
        out = df.copy()

        if parsed.specialty_targets:
            pri_match = out["Pri_spec_norm"].isin(parsed.specialty_targets)
            sec_match = pd.Series(False, index=out.index)
            for target in parsed.specialty_targets:
                sec_match |= out["Sec_spec_all_norm"].fillna("").str.contains(re.escape(target), na=False, regex=True)
            specialty_strength = np.where(pri_match, 1.0, np.where(sec_match, 0.9, 0.0))
        else:
            specialty_strength = np.where(out["women_specific_flag"] == 1, 1.0, np.where(out["women_relevant_flag"] == 1, 0.6, 0.0))

        if parsed.target_state:
            state_strength = np.where(out["state_norm"] == parsed.target_state, 1.0, 0.0)
        else:
            state_strength = np.ones(len(out), dtype=float) * 0.5

        tele_strength = out["Telehlth_flag"].astype(float).values if parsed.wants_telehealth else np.ones(len(out), dtype=float) * 0.5
        aff_strength = (
            0.6 * out["ind_assgn_flag"].astype(float).values + 0.4 * out["grp_assgn_flag"].astype(float).values
        ) if parsed.wants_affordable else np.zeros(len(out), dtype=float)
        female_strength = out["female_provider_flag"].astype(float).values if parsed.wants_female_provider else np.zeros(len(out), dtype=float)

        preference_strength = np.maximum.reduce([tele_strength, aff_strength, female_strength])
        intent_strength = np.maximum(out["women_focus_score"].fillna(0).astype(float).values, specialty_strength)
        if parsed.query_intent == "general":
            intent_strength = specialty_strength

        out["relevance_label"] = (
            (intent_strength >= 0.6)
            & ((state_strength >= 1.0) | (parsed.target_state is None))
            & ((preference_strength >= 0.5) | (not (parsed.wants_telehealth or parsed.wants_affordable or parsed.wants_female_provider)))
        ).astype(int)

        out["graded_relevance"] = (
            2.2 * intent_strength
            + 0.9 * state_strength
            + 0.5 * preference_strength
            + 0.5 * out["quality_score"].fillna(0).astype(float).values
            + 0.4 * out["experience_score"].fillna(0).astype(float).values
        )
        return out


class Evaluator:
    @staticmethod
    def precision_at_k(relevances, k):
        rel = relevances[:k]
        return sum(rel) / len(rel) if rel else 0.0

    @staticmethod
    def recall_at_k(relevances, total_relevant, k):
        if total_relevant <= 0:
            return 0.0
        return sum(relevances[:k]) / total_relevant

    @staticmethod
    def ndcg_at_k(gains, k):
        actual = 0.0
        for rank, gain in enumerate(gains[:k]):
            actual += gain / math.log2(rank + 2)
        ideal = 0.0
        for rank, gain in enumerate(sorted(gains, reverse=True)[:k]):
            ideal += gain / math.log2(rank + 2)
        return actual / ideal if ideal > 0 else 0.0

    @staticmethod
    def representation_at_k(df, k, threshold: float = 0.5):
        subset = df.head(k)
        return float((subset["women_focus_score"] >= threshold).mean()) if not subset.empty else 0.0

    @staticmethod
    def corpus_representation(df, threshold: float = 0.5):
        return float((df["women_focus_score"] >= threshold).mean()) if not df.empty else 0.0

    @staticmethod
    def exposure_per_group(df, k, threshold: float = 0.5):
        subset = df.head(k).copy()
        if subset.empty:
            return {"group_exposure": 0.0, "non_group_exposure": 0.0}

        subset["discount"] = [reciprocal_log_discount(i) for i in range(len(subset))]
        group_mask = subset["women_focus_score"] >= threshold
        return {
            "group_exposure": float(subset.loc[group_mask, "discount"].sum()),
            "non_group_exposure": float(subset.loc[~group_mask, "discount"].sum()),
        }





def evaluate_single_ranking(provider_df, query, ranking_df, k, ranking_name):
    labeled_corpus = Labeler.weak_label(provider_df, query)
    labeled_ranking = Labeler.weak_label(ranking_df, query)
    total_relevant = int(labeled_corpus["relevance_label"].sum())
    evaluator = Evaluator()
    exposure = evaluator.exposure_per_group(ranking_df, k)
    return {
        "ranking": ranking_name,
        "precision@k": evaluator.precision_at_k(labeled_ranking["relevance_label"].tolist(), k),
        "recall@k": evaluator.recall_at_k(labeled_ranking["relevance_label"].tolist(), total_relevant, k),
        "ndcg@k": evaluator.ndcg_at_k(labeled_ranking["graded_relevance"].tolist(), k),
        "representation@k": evaluator.representation_at_k(ranking_df, k),
        "corpus_representation": evaluator.corpus_representation(provider_df),
        "group_exposure": exposure["group_exposure"],
        "non_group_exposure": exposure["non_group_exposure"],
        "exposure_gap": exposure["group_exposure"] - exposure["non_group_exposure"],
    }



def evaluate_rankings(provider_df, query, rankings, k):
    rows = [evaluate_single_ranking(provider_df, query, ranking_df, k, ranking_name) for ranking_name, ranking_df in rankings.items()]
    return pd.DataFrame(rows)



def make_output_table(df, score_col):
    cols = [
        "NPI",
        "Provider First Name",
        "Provider Last Name",
        "Pri_spec",
        "City/Town",
        "State",
        "Telehlth_flag",
        "ind_assgn_flag",
        "grp_assgn_flag",
        "female_provider_flag",
        "women_specific_flag",
        "women_relevant_flag",
        "experience_score",
        "quality_score",
        "women_focus_score",
        "semantic_score",
        "specialty_match",
        "location_match",
        "telehealth_pref_score",
        "affordable_pref_score",
        "female_pref_score",
        score_col,
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out



def get_reranker(name):
    normalized = name.lower()
    if normalized in {"linear", "fair", "linear_fairness"}:
        return LinearFairnessReranker(
            target_min_proportion=DEFAULT_FAIRNESS_PARAMS["target_min_proportion"],
            fairness_bonus=DEFAULT_FAIRNESS_PARAMS["bonus"],
            city_diversity_bonus=DEFAULT_FAIRNESS_PARAMS["city_diversity_bonus"],
            specialty_diversity_bonus=DEFAULT_FAIRNESS_PARAMS["specialty_diversity_bonus"],
            female_bonus=DEFAULT_FAIRNESS_PARAMS["female_bonus"],
        )
    if normalized in {"mmr", "mmr_fairness"}:
        return MMRReranker(
            lambda_relevance=DEFAULT_MMR_PARAMS["lambda_relevance"],
            women_boost=DEFAULT_MMR_PARAMS["women_boost"],
            specialty_diversity_weight=DEFAULT_MMR_PARAMS["specialty_diversity_weight"],
            city_diversity_weight=DEFAULT_MMR_PARAMS["city_diversity_weight"],
        )
    if normalized in {"fair_top_k", "fairtopk", "fa_ir", "fa*ir", "fair-k"}:
        return FAIRTopKReranker(
            target_protected_proportion=DEFAULT_FAIR_TOPK_PARAMS["target_protected_proportion"],
            protected_threshold=DEFAULT_FAIR_TOPK_PARAMS["protected_threshold"],
            pool_multiplier=DEFAULT_FAIR_TOPK_PARAMS["pool_multiplier"],
            min_pool_size=DEFAULT_FAIR_TOPK_PARAMS["min_pool_size"],
        )
    raise ValueError(f"Unknown reranker: {name}")




def load_experiment_queries(queries_file: Optional[str] = None):
    """Load one query per line from a text file; fall back to built-in defaults."""
    candidates = []
    if queries_file:
        candidates.append(Path(queries_file))
    candidates.append(Path(DEFAULT_QUERIES_FILE))
    candidates.append(Path(__file__).resolve().parent / DEFAULT_QUERIES_FILE)

    for candidate in candidates:
        if candidate and candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                loaded = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
            if loaded:
                return loaded
    return list(DEFAULT_EXPERIMENT_QUERIES)




# main workflow 
def build_and_persist_index(data_dir, index_dir, save_full_provider_table: bool = False):
    data_path = Path(data_dir)
    index_path = Path(index_dir)
    validate_input_files(data_dir)

    repo = DataRepository(data_path)
    feature_builder = FeatureBuilder(repo)
    provider_df = feature_builder.build_provider_table()

    index = PersistentProviderIndex(index_path)
    index.build(provider_df)

    if save_full_provider_table:
        index_path.mkdir(parents=True, exist_ok=True)
        provider_df.to_csv(index_path / "provider_feature_table.csv", index=False)

    print(f"Built persistent index at: {index.index_path}")
    print(f"Providers indexed: {len(provider_df)}")
    if save_full_provider_table:
        print(index_path / "provider_feature_table.csv")



def search_with_index(index_dir, query, out_dir, top_k: int = 20, reranker_name: str = "linear"):
    index_path = Path(index_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    index = PersistentProviderIndex(index_path)
    payload = index.load()

    provider_df = payload["provider_df"]
    vectorizer = payload["vectorizer"]
    matrix = payload["matrix"]

    ranker = SemanticRanker(provider_df, vectorizer, matrix)
    baseline_pool, tier_used, parsed = ranker.search(query=query, top_k=max(top_k * 5, 100))
    baseline_topk = baseline_pool.head(top_k).copy()

    linear_reranker = get_reranker("linear")
    mmr_reranker = get_reranker("mmr")
    fair_topk_reranker = get_reranker("fair_top_k")
    chosen_reranker = get_reranker(reranker_name)

    linear_topk = linear_reranker.rerank(baseline_pool, top_k=top_k)
    mmr_topk = mmr_reranker.rerank(baseline_pool, top_k=top_k)
    fair_topk = fair_topk_reranker.rerank(baseline_pool, top_k=top_k)
    chosen_topk = chosen_reranker.rerank(baseline_pool, top_k=top_k)

    ranking_map = {
        "baseline": baseline_topk,
        "linear_fairness": linear_topk,
        "mmr_fairness": mmr_topk,
        "fair_top_k": fair_topk,
    }
    metrics_df = evaluate_rankings(provider_df, query, ranking_map, top_k)

    make_output_table(baseline_topk, "baseline_score").to_csv(out_path / "baseline_ranking.csv", index=False)
    make_output_table(linear_topk, "fair_score").to_csv(out_path / "linear_fair_reranked_ranking.csv", index=False)
    make_output_table(mmr_topk, "fair_score").to_csv(out_path / "mmr_fair_reranked_ranking.csv", index=False)
    make_output_table(fair_topk, "fair_score").to_csv(out_path / "fair_top_k_reranked_ranking.csv", index=False)
    make_output_table(chosen_topk, "fair_score").to_csv(out_path / "selected_reranked_ranking.csv", index=False)
    metrics_df.to_csv(out_path / "ranking_metrics.csv", index=False)

    print("Parsed query:")
    print(f"- query_intent: {parsed.query_intent}")
    print(f"- target_state: {parsed.target_state}")
    print(f"- target_city: {parsed.target_city}")
    print(f"- specialty_targets: {parsed.specialty_targets}")
    print(f"- wants_telehealth: {parsed.wants_telehealth}")
    print(f"- wants_affordable: {parsed.wants_affordable}")
    print(f"- wants_female_provider: {parsed.wants_female_provider}")
    print(f"- wants_quality: {parsed.wants_quality}")
    print(f"- wants_experience: {parsed.wants_experience}")
    print(f"- retrieval_tier_used: {tier_used}")
    print()
    print("Saved:")
    print(out_path / "baseline_ranking.csv")
    print(out_path / "linear_fair_reranked_ranking.csv")
    print(out_path / "mmr_fair_reranked_ranking.csv")
    print(out_path / "fair_top_k_reranked_ranking.csv")
    print(out_path / "selected_reranked_ranking.csv")
    print(out_path / "ranking_metrics.csv")
    print()
    print("Metrics:")
    print(metrics_df.to_string(index=False))



def run_experiments(index_dir, out_dir, top_k: int = 20, queries: Optional[List[str]] = None):
    index = PersistentProviderIndex(Path(index_dir))
    payload = index.load()
    provider_df = payload["provider_df"]
    vectorizer = payload["vectorizer"]
    matrix = payload["matrix"]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ranker = SemanticRanker(provider_df, vectorizer, matrix)
    linear_reranker = get_reranker("linear")
    mmr_reranker = get_reranker("mmr")
    fair_topk_reranker = get_reranker("fair_top_k")

    queries = queries or load_experiment_queries(None)
    all_metrics: List[pd.DataFrame] = []

    for idx, query in enumerate(queries, start=1):
        baseline_pool, tier_used, parsed = ranker.search(query=query, top_k=max(top_k * 5, 100))
        baseline_topk = baseline_pool.head(top_k).copy()
        linear_topk = linear_reranker.rerank(baseline_pool, top_k=top_k)
        mmr_topk = mmr_reranker.rerank(baseline_pool, top_k=top_k)
        fair_topk = fair_topk_reranker.rerank(baseline_pool, top_k=top_k)

        metrics_df = evaluate_rankings(
            provider_df,
            query,
            {
                "baseline": baseline_topk,
                "linear_fairness": linear_topk,
                "mmr_fairness": mmr_topk,
                "fair_top_k": fair_topk,
            },
            top_k,
        )
        metrics_df.insert(0, "query", query)
        metrics_df.insert(1, "query_intent", parsed.query_intent)
        metrics_df.insert(2, "retrieval_tier_used", tier_used)
        all_metrics.append(metrics_df)

        query_dir = out_path / f"query_{idx:02d}"
        query_dir.mkdir(parents=True, exist_ok=True)
        make_output_table(baseline_topk, "baseline_score").to_csv(query_dir / "baseline_ranking.csv", index=False)
        make_output_table(linear_topk, "fair_score").to_csv(query_dir / "linear_fair_reranked_ranking.csv", index=False)
        make_output_table(mmr_topk, "fair_score").to_csv(query_dir / "mmr_fair_reranked_ranking.csv", index=False)
        make_output_table(fair_topk, "fair_score").to_csv(query_dir / "fair_top_k_reranked_ranking.csv", index=False)
        metrics_df.to_csv(query_dir / "ranking_metrics.csv", index=False)

    combined = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    combined.to_csv(out_path / "all_query_metrics.csv", index=False)
    if not combined.empty:
        summary = combined.groupby("ranking", as_index=False).agg(
            query_count=("query", "nunique"),
            mean_precision_at_k=("precision@k", "mean"),
            std_precision_at_k=("precision@k", "std"),
            mean_recall_at_k=("recall@k", "mean"),
            mean_ndcg_at_k=("ndcg@k", "mean"),
            std_ndcg_at_k=("ndcg@k", "std"),
            mean_representation_at_k=("representation@k", "mean"),
            mean_group_exposure=("group_exposure", "mean"),
            mean_non_group_exposure=("non_group_exposure", "mean"),
            mean_exposure_gap=("exposure_gap", "mean"),
        )
        summary = summary.fillna(0.0)
        summary.to_csv(out_path / "summary_metrics.csv", index=False)

        intent_summary = combined.groupby(["query_intent", "ranking"], as_index=False).agg(
            query_count=("query", "nunique"),
            mean_precision_at_k=("precision@k", "mean"),
            mean_recall_at_k=("recall@k", "mean"),
            mean_ndcg_at_k=("ndcg@k", "mean"),
            mean_representation_at_k=("representation@k", "mean"),
            mean_exposure_gap=("exposure_gap", "mean"),
        )
        intent_summary.to_csv(out_path / "summary_metrics_by_intent.csv", index=False)

        print(f"Ran {combined['query'].nunique()} queries.")
        print(summary.to_string(index=False))
        print(out_path / "summary_metrics.csv")
        print(out_path / "summary_metrics_by_intent.csv")
    print(out_path / "all_query_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Women’s healthcare provider IR system with stronger data integration and multiple fairness-aware rerankers")
    parser.add_argument("--mode", choices=["build_index", "search", "run_experiments"], default="search")
    parser.add_argument("--data_dir", type=str, default=".", help="Folder containing the CSV files")
    parser.add_argument("--index_dir", type=str, default=INDEX_DIRNAME, help="Folder where the persistent index is stored")
    parser.add_argument("--query", type=str, default=EXAMPLE_QUERY, help="Natural-language query")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k cutoff")
    parser.add_argument("--reranker", choices=["linear", "mmr", "fair_top_k"], default="linear", help="Which reranker to use for selected_reranked_ranking.csv")
    parser.add_argument("--save_full_provider_table", action="store_true", help="Save provider feature table when building the index")
    parser.add_argument("--queries_file", type=str, default="", help="Optional text file containing one query per line for experiments")
    args = parser.parse_args()

    if args.mode == "build_index":
        build_and_persist_index(args.data_dir, args.index_dir, args.save_full_provider_table)
    elif args.mode == "search":
        search_with_index(args.index_dir, args.query, args.out_dir, args.top_k, args.reranker)
    else:
        experiment_queries = load_experiment_queries(args.queries_file)
        run_experiments(args.index_dir, args.out_dir, args.top_k, experiment_queries)
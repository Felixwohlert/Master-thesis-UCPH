"""
Build cohort weight matrix and L_supply time series from demographics.xlsx.

The Excel file is expected to have a single sheet with:
  - Row 1  : [None, year1, year2, ...]   (integer year column headers)
  - Row 2+ : ['X ar', count1, count2, ...]

L_supply is computed as the working-age population (ages labor_age_min to the
statutory retirement age, inclusive) in each year, using the Danish
retirement-age reform schedule defined in RETIREMENT_AGE_SCHEDULE below.
"""

import argparse
import csv
import os
import re

import numpy as np



# ---------------------------------------------------------------------------
# xlsx reader
# ---------------------------------------------------------------------------
def read_demographics_xlsx(xlsx_path):
    """
    Read demographics.xlsx and return raw arrays.

    Returns
    -------
    years : list of int       -- calendar years (column headers)
    ages  : list of int       -- age values parsed from row labels
    data  : np.ndarray (n_ages, n_years), dtype float
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError(
            "openpyxl is required to read demographics.xlsx. "
            "Install with:  pip install openpyxl"
        )

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))

    # Header row: find year columns
    year_cols = []
    for ci, val in enumerate(rows[0]):
        if val is None:
            continue
        try:
            year_cols.append((ci, int(val)))
        except (ValueError, TypeError):
            pass

    years   = [yr for _, yr in year_cols]
    col_idx = [ci for ci, _ in year_cols]

    # Data rows: parse age labels (e.g. "16 ar")
    age_pattern = re.compile(r"^(\d+)")
    ages      = []
    data_rows = []

    for row in rows[1:]:
        label = row[0]
        if label is None:
            continue
        m = age_pattern.match(str(label).strip())
        if not m:
            continue
        age = int(m.group(1))
        vals = []
        for ci in col_idx:
            raw = row[ci] if ci < len(row) else None
            try:
                vals.append(float(raw) if raw is not None else 0.0)
            except (ValueError, TypeError):
                vals.append(0.0)
        ages.append(age)
        data_rows.append(vals)

    data = np.array(data_rows, dtype=float)
    return years, ages, data




# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_cohort_weights_from_xlsx(xlsx_path, J=80, labor_age_min=16):
    """
    Build cohort weight matrix and L_supply from demographics.xlsx.

    Parameters
    ----------
    xlsx_path          : path to demographics.xlsx
    J                  : number of age bins (ages 16 to 15+J)
    labor_age_min      : minimum age included in L_supply

    L_supply for year t sums population in ages
        labor_age_min ... retirement_age(year_t)  inclusive.

    Returns
    -------
    years              : list of int
    people_matrix      : np.ndarray (T, J)
    cohort_weights     : np.ndarray (T, J)   row-normalised
    labor_force_people : np.ndarray (T,)
    labor_force_norm   : np.ndarray (T,)     normalised to year 0 = 1.0
    """
    years_raw, ages_raw, data = read_demographics_xlsx(xlsx_path)
    T = len(years_raw)
    age_min = 16

    # People matrix (data has shape n_ages x T)
    people_matrix = np.zeros((T, J), dtype=float)
    for i, age in enumerate(ages_raw):
        if age_min <= age < age_min + J:
            people_matrix[:, age - age_min] += data[i, :]

    # Cohort weights (row-normalised)
    row_sums = people_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    cohort_weights = people_matrix / row_sums



    return years_raw, people_matrix, cohort_weights


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build cohort weight matrix from demographics.xlsx."
    )
    parser.add_argument("--xlsx",          type=str, default="demographics.xlsx")
    parser.add_argument("--J",             type=int, default=80)
    parser.add_argument("--labor-age-min", type=int, default=16)
    parser.add_argument("--out-prefix",    type=str, default="cohort_weights")
    parser.add_argument("--folder",        type=str, default=".")
    args = parser.parse_args()

    xlsx_path = os.path.join(args.folder, args.xlsx)

    years, people_matrix, cohort_weights, labor_force_people, labor_force_norm = (
        build_cohort_weights_from_xlsx(
            xlsx_path, J=args.J, labor_age_min=args.labor_age_min,
        )
    )

    prefix = os.path.join(args.folder, args.out_prefix)

    out_npy       = f"{prefix}.npy"
    out_labor_npy = f"{prefix}_labor_force_{args.labor_age_min}_ret_norm.npy"
    np.save(out_npy,       cohort_weights)
    np.save(out_labor_npy, labor_force_norm)

    out_csv = f"{prefix}.csv"
    header  = ["YEAR"] + [f"age_{j}" for j in range(args.J)]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for t, y in enumerate(years):
            w.writerow([y] + cohort_weights[t].tolist())

    out_people_csv = f"{prefix}_people.csv"
    with open(out_people_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for t, y in enumerate(years):
            w.writerow([y] + people_matrix[t].tolist())

    row_sums = cohort_weights.sum(axis=1)
    print(f"Built cohort matrix shape : {cohort_weights.shape}")
    print(f"Year range                : {years[0]}-{years[-1]}")
    print(f"Row sums min/max          : {row_sums.min():.12f} / {row_sums.max():.12f}")
    print(f"Saved: {out_npy}, {out_csv}, {out_people_csv}")


if __name__ == "__main__":
    main()

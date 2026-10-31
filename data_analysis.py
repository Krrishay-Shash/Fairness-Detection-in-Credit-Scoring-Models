"""
HMDA + Census ACS Data Pull & Join
-----------------------------------
Step 1: Pull HMDA mortgage application data (free, no key needed) for one state/year.
Step 2: Pull Census ACS 5-Year tract-level rent-burden data (needs your free API key).
Step 3: Join them on census tract so each mortgage application gets a
        neighborhood-level "alternative data" proxy (rent burden %).

Before running:
  pip install pandas requests --break-system-packages

Edit the CONFIG section below, then run:
  python pull_and_join_data.py
"""

import requests
import pandas as pd
import io

# ============ CONFIG - edit these ============
CENSUS_API_KEY = "e2fa3c793c7dd3f60624c5e45710133f62509952"
STATE_ABBR = "IL"          # two-letter state code for HMDA (e.g. "IL", "TX", "CA")
STATE_FIPS = "17"          # matching FIPS code for Census API (IL=17, TX=48, CA=06)
YEAR = 2023                # HMDA year: 2018-2025 available
# ================================================


def pull_hmda_data(state_abbr: str, year: int) -> pd.DataFrame:
    """
    Downloads raw HMDA loan-level data for one state/year, filtered to
    originated (1) and denied (3) applications only, so our target
    variable (approved vs denied) is clean and binary.
    """
    url = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
    params = {
        "states": state_abbr,
        "years": year,
        "actions_taken": "1,3",   # 1 = Loan originated, 3 = Application denied
    }
    print(f"Requesting HMDA data for {state_abbr}, {year} ... this can take a minute.")
    resp = requests.get(url, params=params, timeout=300)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    print(f"Pulled {len(df):,} HMDA records.")
    return df


def pull_census_rent_burden(state_fips: str, api_key: str, year: int = 2022) -> pd.DataFrame:
    """
    Pulls tract-level rent burden data from ACS 5-Year Estimates (table B25070).
    B25070_001E = total renter households
    B25070_010E = renters paying 50%+ of income on rent (severely burdened)
    Returns a tract-level GEOID plus a computed rent_burden_pct column.
    """
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "B25070_001E,B25070_010E,B19013_001E",  # +median household income
        "for": "tract:*",
        "in": f"state:{state_fips}",
        "key": api_key,
    }
    print("Requesting Census ACS tract data ...")
    resp = requests.get(base_url, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data[1:], columns=data[0])

    # Build the 11-digit tract GEOID to match HMDA's census_tract format
    df["GEOID"] = df["state"] + df["county"] + df["tract"]

    for col in ["B25070_001E", "B25070_010E", "B19013_001E"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rent_burden_pct"] = (df["B25070_010E"] / df["B25070_001E"]) * 100
    df = df.rename(columns={"B19013_001E": "median_household_income"})
    print(f"Pulled {len(df):,} census tracts.")
    return df[["GEOID", "rent_burden_pct", "median_household_income"]]


def join_datasets(hmda_df: pd.DataFrame, census_df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins HMDA records to census tract data on the tract GEOID.
    HMDA's census_tract column should already be an 11-digit FIPS code (state+county+tract).
    """
    hmda_df["census_tract"] = hmda_df["census_tract"].astype(str).str.zfill(11)
    census_df["GEOID"] = census_df["GEOID"].astype(str).str.zfill(11)

    merged = hmda_df.merge(census_df, left_on="census_tract", right_on="GEOID", how="left")
    matched = merged["rent_burden_pct"].notna().sum()
    print(f"Joined {matched:,} / {len(merged):,} HMDA records to a census tract "
          f"({matched/len(merged):.1%} match rate).")
    return merged


if __name__ == "__main__":
    hmda = pull_hmda_data(STATE_ABBR, YEAR)
    census = pull_census_rent_burden(STATE_FIPS, CENSUS_API_KEY)
    final = join_datasets(hmda, census)

    out_path = "hmda_census_joined.csv"
    final.to_csv(out_path, index=False)
    print(f"Saved joined dataset to {out_path} ({len(final):,} rows, {len(final.columns)} columns).")


    









    # ============================================================
# Figures for "Who Pays for the Fix?"
# Reproduces all four paper figures in R using ggplot2.
#
# Install once:
#   install.packages(c("ggplot2", "dplyr", "patchwork"))
#
# Run:
#   source("make_figures.R")
#
# Outputs (written to working directory):
#   fig1_pipeline.png, fig2_fairness.png,
#   fig3_whopays.png,  fig4_selection_rates.png
# ============================================================

library(ggplot2)
library(dplyr)
library(patchwork)

# ---- shared palette (matches the paper) ----
BLUE   <- "#4472C4"
ORANGE <- "#ED7D31"
RED    <- "#C00000"
GREEN  <- "#548235"
GREY   <- "#B0B0B0"

paper_theme <- theme_minimal(base_size = 9) +
  theme(
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(colour = "#DDDDDD", linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    axis.title.x       = element_blank(),
    plot.title         = element_text(size = 8.5, hjust = 0.5),
    legend.position    = "bottom",
    legend.title       = element_blank(),
    strip.text         = element_text(size = 8.5)
  )


# ============================================================
# FIGURE 2 - Fairness metrics, Model A vs Model B
# ============================================================
fairness <- data.frame(
  model = rep(c("Logistic\nRegression", "Random\nForest", "XGBoost"), times = 6),
  featureset = rep(rep(c("Model A (baseline)", "Model B (alt-data)"), each = 3), times = 3),
  metric = rep(c("Disparate impact ratio\n(higher = fairer)",
                 "Demographic parity diff.\n(lower = fairer)",
                 "Equalized odds diff.\n(lower = fairer)"), each = 6),
  value = c(
    0.855, 0.835, 0.821,   0.869, 0.825, 0.822,   # disparate impact
    0.136, 0.139, 0.162,   0.123, 0.155, 0.161,   # demographic parity
    0.126, 0.057, 0.155,   0.099, 0.119, 0.154    # equalized odds
  ),
  stringsAsFactors = FALSE
)

fairness$metric <- factor(fairness$metric, levels = unique(fairness$metric))
fairness$model  <- factor(fairness$model,
                          levels = c("Logistic\nRegression", "Random\nForest", "XGBoost"))

fig2 <- ggplot(fairness, aes(x = model, y = value, fill = featureset)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(aes(label = sprintf("%.3f", value)),
            position = position_dodge(width = 0.8),
            vjust = -0.5, size = 2.1) +
  facet_wrap(~metric, scales = "free_y") +
  scale_fill_manual(values = c("Model A (baseline)" = BLUE,
                                "Model B (alt-data)" = ORANGE)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.20))) +
  labs(y = NULL) +
  paper_theme +
  theme(axis.text.x = element_text(size = 7.5))

ggsave("fig2_fairness.png", fig2, width = 9.2, height = 3.2, dpi = 200)


# ============================================================
# FIGURE 3 - Net change in approvals by group ("who pays")
# ============================================================
group_levels <- c("White", "Black or\nAfrican American", "Asian",
                  "Amer. Indian /\nAlaska Native", "2+ minority\nraces",
                  "Native Hawaiian /\nPacific Islander")

flips <- data.frame(
  group = factor(rep(group_levels, times = 2), levels = rev(group_levels)),
  model = rep(c("XGBoost: aggregate fairness unchanged,\nbut losses fall unevenly",
                "Logistic Regression: aggregate fairness improved,\nyet groups move in opposite directions"),
              each = 6),
  net = c(-33, -86, -11, -2, -2, 0,      # XGBoost
           21, -24,  10, -1, -2, 1),     # Logistic Regression
  stringsAsFactors = FALSE
)

flips$model <- factor(flips$model, levels = unique(flips$model))
flips$direction <- ifelse(flips$net < 0, "loss", ifelse(flips$net > 0, "gain", "none"))

fig3 <- ggplot(flips, aes(x = net, y = group, fill = direction)) +
  geom_col(width = 0.62) +
  geom_vline(xintercept = 0, colour = "#333333", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%+d", net),
                hjust = ifelse(net >= 0, -0.25, 1.25)),
            size = 2.4) +
  facet_wrap(~model, scales = "free_x") +
  scale_fill_manual(values = c(loss = RED, gain = GREEN, none = GREY),
                    guide = "none") +
  scale_x_continuous(expand = expansion(mult = 0.22)) +
  labs(x = "Net change in approvals (Model A -> Model B)", y = NULL) +
  paper_theme +
  theme(
    axis.title.x       = element_text(size = 8),
    axis.text.y        = element_text(size = 7.5),
    panel.grid.major.x = element_line(colour = "#DDDDDD", linewidth = 0.3),
    panel.grid.major.y = element_blank()
  )

ggsave("fig3_whopays.png", fig3, width = 9.2, height = 3.3, dpi = 200)


# ============================================================
# FIGURE 4 - Per-group selection rates (metric decomposition)
# ============================================================
sel_levels <- c("Native Hawaiian /\nPacific Islander", "2+ minority\nraces",
                "Amer. Indian /\nAlaska Native", "Black or\nAfrican American",
                "Asian", "White")

selection <- data.frame(
  group = factor(sel_levels, levels = sel_levels),
  rate  = c(0.810811, 0.816794, 0.850000, 0.873880, 0.929175, 0.933555),
  n     = c("~75", "~121", "~333", "~5,930", "~3,726", "~40,311"),
  role  = c("extreme", "other", "other", "black", "other", "extreme"),
  stringsAsFactors = FALSE
)

# numeric y positions for the annotation brackets (factor levels map to 1..6)
y_nhpi  <- 1
y_black <- 4
y_white <- 6

fig4 <- ggplot(selection, aes(x = rate, y = group, fill = role)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = sprintf("%.3f  (n = %s)", rate, n)),
            hjust = -0.08, size = 2.4) +
  scale_fill_manual(values = c(extreme = RED, black = BLUE, other = GREY),
                    guide = "none") +
  # bracket: Black-White comparison
  annotate("segment", x = 1.055, xend = 1.055, y = y_black, yend = y_white,
           colour = BLUE, linewidth = 0.45,
           arrow = arrow(ends = "both", length = unit(0.06, "inches"))) +
  annotate("text", x = 1.065, y = (y_black + y_white) / 2,
           label = "Black-White ratio\nalone: 0.936",
           colour = BLUE, size = 2.3, hjust = 0, lineheight = 1.1) +
  # bracket: the two groups that actually set the metric
  annotate("segment", x = 1.175, xend = 1.175, y = y_nhpi, yend = y_white,
           colour = RED, linewidth = 0.5,
           arrow = arrow(ends = "both", length = unit(0.06, "inches"))) +
  annotate("text", x = 1.185, y = (y_nhpi + y_white) / 2 - 0.4,
           label = "These two groups alone set\nthe reported metrics:\n0.811 / 0.934 = 0.869",
           colour = RED, size = 2.3, hjust = 0, lineheight = 1.15) +
  scale_x_continuous(limits = c(0.70, 1.40),
                      breaks = seq(0.70, 1.00, by = 0.05),
                      expand = c(0, 0)) +
  labs(x = "Model B selection rate (approval rate), Logistic Regression", y = NULL) +
  paper_theme +
  theme(
    axis.title.x       = element_text(size = 8.5),
    axis.text.y        = element_text(size = 7.5),
    panel.grid.major.x = element_line(colour = "#DDDDDD", linewidth = 0.3),
    panel.grid.major.y = element_blank()
  )

ggsave("fig4_selection_rates.png", fig4, width = 8.8, height = 3.2, dpi = 200)


# ============================================================
# FIGURE 1 - Data and modeling pipeline diagram
# Drawn with rectangles and arrows on a blank coordinate space.
# ============================================================
boxes <- data.frame(
  xmin = c(1, 1, 28, 51, 51, 75),
  xmax = c(22, 22, 45, 70, 70, 98),
  ymin = c(27, 5, 16, 27, 5, 16),
  ymax = c(38, 16, 27, 38, 16, 27),
  fill = c("#DEEBF7", "#FCE4D6", "#EDEDED", "#DEEBF7", "#FCE4D6", "#E2EFDA"),
  colour = c(BLUE, ORANGE, "#666666", BLUE, ORANGE, GREEN),
  label = c(
    "HMDA 2023 (Illinois)\nloan applications\nrace, income, DTI,\nloan amount, tract",
    "Census ACS 5-Year\ntract indicators\nrent burden %,\nmedian income",
    "Join on\n11-digit tract\nGEOID\n\nN = 201,981",
    "Model A (baseline)\nincome, loan amount,\nDTI",
    "Model B (alt-data)\nbaseline + rent burden,\nmedian income",
    "Logistic Regression\nRandom Forest\nXGBoost\n-> AUC + fairness metrics"
  ),
  stringsAsFactors = FALSE
)

arrows_df <- data.frame(
  x    = c(22, 22, 45, 45, 70, 70),
  xend = c(28, 28, 51, 51, 75, 75),
  y    = c(32, 10, 24, 19, 32, 11),
  yend = c(25, 19, 32, 11, 25, 19)
)

fig1 <- ggplot() +
  geom_rect(data = boxes,
            aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
            fill = boxes$fill, colour = boxes$colour, linewidth = 0.5) +
  geom_text(data = boxes,
            aes(x = (xmin + xmax) / 2, y = (ymin + ymax) / 2, label = label),
            size = 2.3, lineheight = 1.15) +
  geom_segment(data = arrows_df,
               aes(x = x, xend = xend, y = y, yend = yend),
               colour = "#555555", linewidth = 0.4,
               arrow = arrow(length = unit(0.07, "inches"), type = "closed")) +
  annotate("text", x = 86.5, y = 11.5,
           label = "Compare A vs B predictions\nper applicant -> \"who pays\" flips",
           size = 2.2, fontface = "italic", colour = "#444444", lineheight = 1.2) +
  geom_segment(aes(x = 86.5, xend = 86.5, y = 16, yend = 14),
               colour = "#555555", linewidth = 0.4,
               arrow = arrow(length = unit(0.07, "inches"), type = "closed")) +
  scale_x_continuous(limits = c(0, 100), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 42), expand = c(0, 0)) +
  theme_void()

ggsave("fig1_pipeline.png", fig1, width = 9.2, height = 2.5, dpi = 200)

cat("All four figures written to:", getwd(), "\n")
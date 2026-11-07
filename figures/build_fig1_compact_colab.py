#!/usr/bin/env python3
"""CREADS Fig. 1 — condensed two-panel version for the paper.

A  Level 1 polygon-tile match  (cropped world map; 3 large zoom insets of the
   poorest, highest-contrast countries embedded in the map's ocean space)
B  Italy: pronounced north-south disparity (zoom)
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, Patch
from matplotlib import font_manager

# ---------------------------------------------------------------- font
for p in ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
    try:
        font_manager.fontManager.addfont(p)
    except Exception:
        pass
avail = {f.name for f in font_manager.fontManager.ttflist}
BASE_FONT = "Liberation Sans" if "Liberation Sans" in avail else "DejaVu Sans"
plt.rcParams.update({"font.family": BASE_FONT, "axes.linewidth": 0.6})

ROOT = os.environ.get("CREADS_ROOT", "/content/creads_repro")
ISOS = ["ARM","AUT","BEL","BGD","BRA","BTN","COL","CUB","EGY","GEO","GUY","IDN","IND","IRN","ITA","JOR","KGZ","KOR","MDG","MEX","NGA","NLD","PNG","THA","TZA","VEN"]

# --- GADM level-1 polygons: auto-download if missing (~915 MB, cached in ROOT) ---
import concurrent.futures as cf, subprocess, zipfile as _zf
GADM_SHP = f"{ROOT}/gadm/shp"
if not all(os.path.isdir(f"{GADM_SHP}/{_i}") for _i in ISOS):
    os.makedirs(GADM_SHP, exist_ok=True)
    def _get(_iso):
        _url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{_iso}_shp.zip"
        _zp = f"{GADM_SHP}/gadm41_{_iso}_shp.zip"
        for _a in range(3):
            try:
                subprocess.run(["curl","-sL","--max-time","900","-o",_zp,_url], check=True)
                _zf.ZipFile(_zp).extractall(f"{GADM_SHP}/{_iso}")
                os.remove(_zp)
                return _iso
            except Exception:
                if _a == 2:
                    raise
    print(f"Downloading GADM level-1 shapefiles for {len(ISOS)} countries (~915 MB)...")
    with cf.ThreadPoolExecutor(6) as _ex:
        list(_ex.map(_get, ISOS))
    print("GADM download complete.")

# ---------------------------------------------------------------- data
matched = pd.read_excel(f"{ROOT}/data/MATCHED_340REGIONS_WITH_GDL_2023.xlsx",
                        usecols=["region_id", "ISO3", "NAME_1", "country_name",
                                 "Subnational_HDI_2023"])
complete = pd.read_excel(f"{ROOT}/data/COMPLETE_584REGIONS_WITH_CONNECTIVITY_2023.xlsx",
                         usecols=["region_id", "ISO3", "NAME_1", "country_name"])
lab_ids = set(matched.region_id)
inv_ids = set(complete.region_id)
hdi_map = dict(zip(matched.region_id, matched.Subnational_HDI_2023))

# ---------------------------------------------------------------- polygons
world = gpd.read_file(f"{ROOT}/geo/ne_110m_admin_0_countries.geojson")
adm = []
for iso in ISOS:
    g = gpd.read_file(f"{ROOT}/gadm/shp/{iso}/gadm41_{iso}_1.shp")
    g["ISO3"] = iso
    adm.append(g)
adm1 = pd.concat(adm, ignore_index=True).to_crs(4326)
adm1["in_inv"] = adm1.GID_1.isin(inv_ids)
adm1["labelled"] = adm1.GID_1.isin(lab_ids)
adm1["hdi"] = adm1.GID_1.map(hdi_map)
miss = matched[~matched.region_id.isin(set(adm1.GID_1))]
if len(miss):
    key = {(r.ISO3, str(r.NAME_1).upper().replace(" ", "")): r.Subnational_HDI_2023
           for r in miss.itertuples()}
    def nm(r):
        return key.get((r.ISO3, str(r.NAME_1).upper().replace(" ", "")))
    hit = adm1.GID_1.isin(lab_ids) | adm1.apply(nm, axis=1).notna()
    adm1.loc[hit, "labelled"] = True
    hh = adm1.apply(nm, axis=1)
    adm1.loc[hh.notna() & ~adm1.GID_1.isin(lab_ids), "hdi"] = hh
print(f"ADM1 features: {len(adm1)} | inventory {adm1.in_inv.sum()} | "
      f"labelled poly {adm1.hdi.notna().sum()} | unmatched {(adm1.in_inv & ~adm1.labelled).sum()}")
p_lab = int(adm1.hdi.notna().sum())
p_unlab = int((adm1.in_inv & ~adm1.labelled).sum())
n_lab, n_unlab = len(matched), len(complete) - len(matched)
share = 100.0 * n_lab / len(complete)
hdi_lo, hdi_hi = matched.Subnational_HDI_2023.min(), matched.Subnational_HDI_2023.max()

labA = adm1[adm1.hdi.notna()]
unlA = adm1[~adm1.labelled]

# ---------------------------------------------------------------- style
cmap_hdi = LinearSegmentedColormap.from_list(
    "hdi_seq", ["#f1e2b6", "#cfe0c3", "#93bcc6", "#5b93b4", "#3b6ba0", "#234a7d", "#183560"])
VMIN_A, VMAX_A = 0.35, 1.0
norm_a = Normalize(VMIN_A, VMAX_A)
VMIN_C, VMAX_C = 0.86, 0.95
norm_c = Normalize(VMIN_C, VMAX_C)

GREY_UNLAB = "#d3d3d3"
BG = "#f2f2f2"
CARD_EDGE = "#b0b0b0"

EXT = dict(lon0=-120, lon1=158, lat0=-28, lat1=56)

# high-contrast zooms: only the poorest, most internally disparate countries
ZOOMS = [
    ("a", "Nigeria & West Africa",
     dict(lon0=2,   lon1=17,  lat0=3.5,  lat1=14.5),
     (0.012, 0.062, 0.145, 0.372)),          # card on SE Pacific (ocean/grey Peru)
    ("b", "Tanzania & East Africa",
     dict(lon0=28,  lon1=42,  lat0=-12,  lat1=0.5),
     (0.325, 0.062, 0.150, 0.368)),          # card on S Atlantic / grey southern Africa
    ("c", "Papua New Guinea",
     dict(lon0=138, lon1=152, lat0=-10,  lat1=0.5),
     (0.878, 0.575, 0.118, 0.310)),          # card on N Pacific / grey Japan
]

fig = plt.figure(figsize=(11.4, 7.7), dpi=300)
gs = fig.add_gridspec(2, 2, height_ratios=[1.65, 1.0], width_ratios=[1.25, 1.0],
                      hspace=0.16, wspace=0.12,
                      left=0.035, right=0.975, top=0.930, bottom=0.105)
axA = fig.add_subplot(gs[0, :])


def draw_maps(ax, labelled_plot=True):
    world.plot(ax=ax, color=BG, edgecolor="#d8d8d8", linewidth=0.35)
    if labelled_plot:
        labA.plot(column="hdi", cmap=cmap_hdi, norm=norm_a, ax=ax,
                  edgecolor="white", linewidth=0.35)
    else:
        labA.plot(ax=ax, color="#2e5f8f", edgecolor="white", linewidth=0.35)
    unlA.plot(ax=ax, color=GREY_UNLAB, edgecolor="white", linewidth=0.35)
    ax.set_xlim(EXT["lon0"], EXT["lon1"])
    ax.set_ylim(EXT["lat0"], EXT["lat1"])
    ax.set_aspect("equal")
    ax.set_facecolor("white")


def panel_title(ax, letter, title, subtitle):
    ax.text(0.0, 1.118, f"{letter}  {title}", transform=ax.transAxes,
            fontsize=13.5, fontweight="bold", va="bottom", ha="left", color="#1a1a1a")
    ax.text(0.0, 1.052, subtitle, transform=ax.transAxes,
            fontsize=9.0, va="bottom", ha="left", color="#555555")


# ================================================================ PANEL A
draw_maps(axA, labelled_plot=True)

fw_in, fh_in = fig.get_figwidth(), fig.get_figheight()
for letter, name, ext, (fx, fy, fw, fh) in ZOOMS:
    axc = axA.inset_axes([fx, fy, fw, fh])
    axc.set_facecolor("white")
    axc.set_xticks([]); axc.set_yticks([])
    for sp in axc.spines.values():
        sp.set_color(CARD_EDGE); sp.set_linewidth(1.1)
    # adapt the data window to the card's real aspect so the map fills it
    dlon, dlat = ext["lon1"] - ext["lon0"], ext["lat1"] - ext["lat0"]
    pos = axc.get_position()
    slot_aspect = (pos.width * fw_in) / (pos.height * fh_in)
    if dlon / dlat < slot_aspect:
        c = (ext["lat0"] + ext["lat1"]) / 2
        dlat = dlon / slot_aspect
        ext = dict(lon0=ext["lon0"], lon1=ext["lon1"],
                   lat0=c - dlat / 2, lat1=c + dlat / 2)
    else:
        c = (ext["lon0"] + ext["lon1"]) / 2
        dlon = dlat * slot_aspect
        ext = dict(lon0=c - dlon / 2, lon1=c + dlon / 2,
                   lat0=ext["lat0"], lat1=ext["lat1"])
    axm = axc.inset_axes([0, 0, 1, 1])
    axm.set_facecolor("white")
    world.plot(ax=axm, color=BG, edgecolor="#d8d8d8", linewidth=0.4)
    labA.plot(column="hdi", cmap=cmap_hdi, norm=norm_a, ax=axm,
              edgecolor="white", linewidth=0.5)
    unlA.plot(ax=axm, color=GREY_UNLAB, edgecolor="white", linewidth=0.5)
    axm.set_xlim(ext["lon0"], ext["lon1"])
    axm.set_ylim(ext["lat0"], ext["lat1"])
    axm.set_aspect("equal")
    axm.set_xticks([]); axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_color("#cccccc"); sp.set_linewidth(0.4)
    axc.text(0.018, 0.975, letter, transform=axc.transAxes, fontsize=9.5,
             fontweight="bold", va="top", ha="left", color="#1a1a1a",
             bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.92))
    axc.text(0.062, 0.975, name, transform=axc.transAxes, fontsize=7.8,
             va="top", ha="left", color="#333333",
             bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.92))
    # dashed extent rectangle + letter tag on the main map
    r = Rectangle((ext["lon0"], ext["lat0"]), ext["lon1"] - ext["lon0"],
                  ext["lat1"] - ext["lat0"], fill=False, ec=CARD_EDGE,
                  lw=0.9, ls=(0, (3, 2)), zorder=6)
    axA.add_patch(r)
    axA.text(ext["lon1"] - 1.0, ext["lat1"] - 0.8, letter, fontsize=8.5,
             fontweight="bold", color="#444444", va="top", ha="right", zorder=7,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))

panel_title(axA, "A", "Level 1 polygon\u2013tile match",
            "Subnational region\u2013tile alignment; polygon-level HDI across "
            f"{n_lab} labelled regions in {matched.ISO3.nunique()} countries \u00b7 "
            "zooms: the poorest, highest-contrast regions")
axA.set_xticks([-100, -50, 0, 50, 100, 150])
axA.set_yticks([-30, -15, 0, 15, 30, 45])
axA.set_xticklabels(["-100", "-50", "0", "50", "100", "150"], fontsize=8)
axA.set_yticklabels(["-30", "-15", "0", "15", "30", "45"], fontsize=8)
axA.set_xlabel("Longitude", fontsize=8.5, labelpad=1.5)
axA.set_ylabel("Latitude", fontsize=8.5, labelpad=2.0)
axA.tick_params(length=2.5, width=0.6)
for sp in axA.spines.values():
    sp.set_color("#aaaaaa")

# ================================================================ ROW 2 LEFT
# inventory & label coverage (absorbs the former footprint panel)
axL = fig.add_subplot(gs[1, 0])
axL.axis("off")
handles = [
    Patch(facecolor="#2e5f8f", edgecolor="white",
          label=f"HDI-matched \u00b7 {p_lab} polygons / {n_lab} regions"),
    Patch(facecolor=GREY_UNLAB, edgecolor="white",
          label=f"No HDI label \u00b7 {p_unlab} polygons / {n_unlab} regions"),
]
axL.text(0.02, 1.02, "Inventory & label coverage", transform=axL.transAxes,
         fontsize=10.5, fontweight="bold", va="bottom", ha="left", color="#1a1a1a")
axL.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.97),
           fontsize=8.6, frameon=False, handlelength=1.2, handleheight=1.1,
           labelspacing=0.6)
caxL = axL.inset_axes([0.03, 0.34, 0.62, 0.13])
sm = plt.cm.ScalarMappable(norm=norm_a, cmap=cmap_hdi)
cb = fig.colorbar(sm, cax=caxL, orientation="horizontal")
cb.set_ticks([0.4, 0.6, 0.8, 1.0])
cb.ax.tick_params(labelsize=7.5, width=0.6, length=2.5)
cb.outline.set_linewidth(0.5)
caxL.set_title("Subnational HDI 2023", fontsize=8.4, pad=3.0, color="#333333")
axL.text(0.03, 0.13,
         f"{complete.ISO3.nunique()} countries \u00b7 {len(complete)} regions in inventory \u00b7 "
         f"{n_lab} with a 2023 HDI label ({share:.1f}%)", transform=axL.transAxes,
         fontsize=8.6, va="top", ha="left", color="#333333")
axL.text(0.03, 0.015,
         f"Labelled-region HDI range {hdi_lo:.3f}\u2013{hdi_hi:.3f}",
         transform=axL.transAxes, fontsize=8.6, va="top", ha="left", color="#555555")

# ================================================================ PANEL B
axB = fig.add_subplot(gs[1, 1])
ita = adm1[adm1.ISO3 == "ITA"]
world.plot(ax=axB, color=BG, edgecolor="#d8d8d8", linewidth=0.5)
ita_lab = ita[ita.hdi.notna()]
ita_unl = ita[~ita.labelled]
ita_lab.plot(column="hdi", cmap=cmap_hdi, norm=norm_c, ax=axB,
             edgecolor="white", linewidth=0.9)
ita_unl.plot(ax=axB, color=GREY_UNLAB, edgecolor="white", linewidth=0.9)
axB.set_xlim(6.4, 18.8)
axB.set_ylim(35.6, 47.4)
axB.set_aspect("equal")
axB.set_xticks([]); axB.set_yticks([])
for sp in axB.spines.values():
    sp.set_color("#aaaaaa")

caxB = axB.inset_axes([1.035, 0.08, 0.055, 0.84])
smc = plt.cm.ScalarMappable(norm=norm_c, cmap=cmap_hdi)
cbB = fig.colorbar(smc, cax=caxB, orientation="vertical")
hmin, hmax = ita_lab.hdi.min(), ita_lab.hdi.max()
cbB.set_ticks([0.87, 0.89, 0.91, 0.93, 0.95])
cbB.ax.tick_params(labelsize=7.5, width=0.6, length=2.5)
cbB.outline.set_linewidth(0.5)
caxB.set_title("HDI", fontsize=8.5, pad=4.0, color="#333333")

panel_title(axB, "B", "Italy: pronounced north\u2013south disparity",
            f"{len(ita)} level-1 regions \u00b7 HDI {hmin:.3f}\u2013{hmax:.3f}")
foot = ("Regions are GADM level-1 units; Trentino\u2013Alto Adige aggregates "
        "the autonomous provinces of Trento and Bolzano.")
axB.text(0.0, -0.045, foot, transform=axB.transAxes, fontsize=7.0,
         color="#4d4d4d", va="top", ha="left")

fig.savefig(f"{ROOT}/fig1_CREADS_compact.png", dpi=300, facecolor="white")
print("SAVED fig1_CREADS_compact.png")
from PIL import Image
Image.open(f"{ROOT}/fig1_CREADS_compact.png").convert("RGB").save(
    f"{ROOT}/fig1_CREADS_compact.pdf", "PDF", resolution=300.0, quality=95)
print("SAVED fig1_CREADS_compact.pdf")

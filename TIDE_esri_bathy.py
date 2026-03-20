import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xarray as xr
from scipy.spatial import cKDTree

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

import cmocean


# define file paths
CSV_FILE = "/Users/auroraczajkowski/Desktop/Ship Time Proposal /Sample Sites new.csv"

# ROMS bathymetry grid file (must contain lon_rho, lat_rho, h)
BATHY_NETCDF = "/Users/auroraczajkowski/Desktop/COAST/GRID_SDTJRE_LV4_mss_oct2024.nc"


# settings 
GROUP_TO_PLOT = "A"

EXTENT_PAD = 0.05
POINT_SIZE = 35
BATHY_ALPHA = 0.6

LEVELS = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]


# DMS parsing function
def parse_dms(dms: str) -> float:
    pattern = r"""^\s*
                  (?P<deg>\d+)°\s*
                  (?P<min>\d+)'?\s*
                  (?P<sec>\d+(?:\.\d+)?)"?
                  (?P<hemi>[NnSsEeWw])
                  \s*$"""
    m = re.match(pattern, dms, re.VERBOSE)
    deg = float(m.group("deg"))
    minu = float(m.group("min"))
    sec = float(m.group("sec"))
    hemi = m.group("hemi").upper()
    dd = deg + minu/60 + sec/3600
    return -dd if hemi in ("S", "W") else dd


# load sites 
df = pd.read_csv(CSV_FILE, dtype=str)

df = df[df["location"].notna() & df["location"].str.strip().ne("")].copy()

split_coords = df["location"].str.extract(
    r"(?P<Latitude>\d+°\d+'[\d.]+\"?[NSns])\s+(?P<Longitude>\d+°\d+'[\d.]+\"?[EWew])"
)

df = df.join(split_coords).dropna(subset=["Latitude", "Longitude"]).copy()

df["lat_dd"] = df["Latitude"].apply(lambda x: parse_dms(str(x).strip()))
df["lon_dd"] = df["Longitude"].apply(lambda x: parse_dms(str(x).strip()))
df["group"] = df["group"].astype(str).str.strip()

sites = df[df["group"] == GROUP_TO_PLOT].copy()

if len(sites) == 0:
    raise ValueError("No Group A sites found.")


# Label column detection
label_col = None
for cand in ["sample site", "site", "station", "id", "name"]:
    if cand in [c.lower() for c in sites.columns]:
        for c in sites.columns:
            if c.lower() == cand:
                label_col = c
                break

# load bathymetry
ds = xr.open_dataset(BATHY_NETCDF, engine="netcdf4", decode_timedelta=False)

lon = ds["lon_rho"].values
lat = ds["lat_rho"].values
depth = ds["h"].values

if "mask_rho" in ds.variables:
    mask = ds["mask_rho"].values
    depth = np.where(mask == 0, np.nan, depth)


# Build spatial index and query nearest bathymetry point for each site
lon_flat = lon.ravel()
lat_flat = lat.ravel()
dep_flat = depth.ravel()

wet = np.isfinite(dep_flat)

coords = np.column_stack((lon_flat[wet], lat_flat[wet]))
tree = cKDTree(coords)

site_coords = np.column_stack((sites["lon_dd"], sites["lat_dd"]))
dist, idx = tree.query(site_coords, k=1)

sites["bathy_depth_m"] = dep_flat[wet][idx]


# Define plot extent with some padding
lat0, lat1 = sites["lat_dd"].min(), sites["lat_dd"].max()
lon0, lon1 = sites["lon_dd"].min(), sites["lon_dd"].max()

extent = (
    lon0 - EXTENT_PAD,
    lon1 + EXTENT_PAD,
    lat0 - EXTENT_PAD,
    lat1 + EXTENT_PAD
)


# Create map with satellite imagery and bathymetry overlay
esri = cimgt.GoogleTiles(
    url=(
        "https://services.arcgisonline.com/ArcGIS/rest/"
        "services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )
)


# plot 
fig = plt.figure(figsize=(7,6))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent(extent, crs=ccrs.PlateCarree())

# Satellite
ax.add_image(esri, 14, zorder=1)

# Bathymetry overlay
bathy = ax.contourf(
    lon, lat, depth,
    levels=LEVELS,
    cmap=cmocean.cm.deep,
    transform=ccrs.PlateCarree(),
    alpha=BATHY_ALPHA,
    zorder=2
)

cb = plt.colorbar(bathy, ax=ax, orientation="vertical", shrink=0.75, pad=0.03)
cb.set_label("Depth (m)")


# Coastline
#ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)

gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                  linewidth=0.4, color="gray", alpha=0.7)
gl.top_labels = False
gl.right_labels = False


# Plot sites + depth labels
for _, row in sites.iterrows():

    ax.scatter(
        row["lon_dd"],
        row["lat_dd"],
        s=POINT_SIZE,
        edgecolor="k",
        facecolor="red",
        transform=ccrs.PlateCarree(),
        zorder=4
    )

    # Site name
    name = ""
    if label_col:
        name = str(row[label_col]).strip()

    depth_text = f"{row['bathy_depth_m']:.1f} m"

    label = f"{name}\n{depth_text}" if name else depth_text

    ax.text(
        row["lon_dd"] + 0.001,
        row["lat_dd"] + 0.001,
        label,
        fontsize=5,
        fontweight="bold",
        transform=ccrs.PlateCarree(),
        zorder=5
    )


ax.set_title("Group A Sites with Depth", fontsize=14)

plt.tight_layout()
plt.show()

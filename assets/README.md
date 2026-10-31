# Offline map data

Runtime map data is stored in `assets/maps/`.

## UN-source world reference map (default)

- Publishers: UN Geospatial and OCHA FISS.
- [Official map service](https://gis.unocha.org/server/rest/services/Hosted/Global_AB_25M_simplified_fs_gray/FeatureServer).
- `un_countries.geojson`: country and territory polygons from layer 18, with 262 source features.
- `un_boundaries.geojson`: coastlines, international boundaries, and other dividing lines from layer 12, with 1,894 source features.
- Coordinates: EPSG:4326 as returned by the service. Original geometry is retained and projected to Web Mercator for display.
- `un_map_sources.json`: source URLs, download time, source usage notes, official line styles, feature counts, and SHA256 checksums. The source is a simplified 1:25 million reference layer suitable for country overviews, not roads or precise boundary measurement.

China and neighboring areas use source geometry, feature codes, and `lbl_en` labels directly. Empty source labels remain empty; internal feature codes are not presented as independent country labels. Land fills do not add uniform solid borders. Boundaries are rendered separately using source line widths and dash patterns so special dividing lines are not displayed as ordinary international boundaries. Display colors are adapted for contrast on the dark map; source resources remain unchanged.

The renderer caches scale-dependent display geometry and reuses a padded viewport while panning. Subpixel details are simplified or omitted in overview displays and return at higher zoom levels; this does not modify the source files. Place labels use their rendered text bounds, spacing, and a viewport-dependent density limit. They hide during dragging and return after release. Online mode uses the basemap's own labels without a duplicate offline label layer.

**Data provenance does not imply territorial endorsement.** As the source notes explain, the boundaries, names, and designations shown do not imply official UN endorsement or acceptance. The original statement is retained in `un_map_sources.json` under `license_info`. This is a UN-source reference map, not a claim of uniquely recognized borders. Map display does not affect training labels, coordinate calculations, or ground-truth evaluation.

Refresh resources manually with the following network-enabled command. Application startup and training do not invoke it:

```powershell
.venv\Scripts\python.exe -m streetgeo refresh-map
```

The tool retrieves all features through pagination, checks totals and duplicate IDs, and obtains line styles from source metadata. Restart the prediction map after refreshing.

## US state boundaries and online roads

`maps/us_states.geojson` contains the 50 US states and DC from the existing Census TIGERweb downloader cache, simplified with a 0.04° tolerance. Source: [Census TIGERweb](https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/0).

When online roads are enabled, the application displays an independent OpenStreetMap basemap and its boundary depiction. It is not labeled as a UN map. Only standard tiles for the current viewport are requested, respecting response cache lifetimes without bulk downloading. See the [OSM tile usage policy](https://operations.osmfoundation.org/policies/tiles/).

The old Natural Earth world outlines have been removed. The default offline map uses the UN-source data described above.

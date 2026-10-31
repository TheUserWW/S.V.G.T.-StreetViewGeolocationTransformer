# StreetGeo prediction client

## Usage

1. Extract the entire folder to a writable location. Keep `best.pt`, `streetgeo`, `assets`, and the other included files.
2. Double-click the `.bat` launcher in this folder.
3. On first launch, the client creates an isolated environment and downloads dependencies. Stay connected and wait for completion. Later launches check and reuse the installed dependencies.
4. Import 1–4 perspective street views of the same location and start prediction. The map shows candidate locations; coordinates can be copied and results exported.

Requires Windows x64 and **64-bit Python 3.10–3.15** with the standard Tcl/Tk components; Python 3.12 is recommended. Inference uses the CPU and does not require an NVIDIA GPU or CUDA. A trained `best.pt` is included, so training data is unnecessary. The launcher chooses PyTorch CPU dependencies according to the Python version.

Images and inference remain local. Offline maps work without network access. Only enabling the optional online road basemap requests OpenStreetMap tiles for the current viewport.

Screenshot calibration accepts a game-map screenshot and at least three corresponding point pairs, then marks a suggested location for manual placement in the game.

## Distribution and updates

Send this complete folder to testers. Do not include training directories, datasets, or the main project's virtual environment. Before redistributing a previously used client, remove its generated `.runtime` folder to exclude dependencies and personal settings. The recipient's launcher recreates the environment. Keep `best.pt` and `assets`.

To replace the model, replace `best.pt` with a trained visual checkpoint. Its architecture is loaded from the checkpoint. `MODEL_INFO.json` describes the bundled default model and is not used for inference.

Dependencies are installed only under `.runtime/envs/` in this folder; system Python packages are not modified. Relocating the client or Python installation creates an environment for the new location.

## Troubleshooting

- Interrupted downloads: run the launcher again to check and finish installing dependencies.
- Startup errors: inspect `.runtime/startup.log` and provide it to the developer.
- Unsupported Python version or architecture: use 64-bit Python 3.12.
- Missing tkinter: add the **Tcl/Tk and IDLE** component through the Python installer.
- `DLL load failed`: check the Microsoft Visual C++ runtime and the detailed startup log.
- Missing `init.tcl`: the launcher locates Tcl/Tk in the base Python installation and includes matching script libraries for 8.6.14/8.6.15. If it still fails, use the Python installer's Modify/Repair option to add Tcl/Tk and IDLE. Keep `.runtime` during an in-place repair to reuse installed model dependencies.

Developers can run `python bootstrap.py --setup-only` to prepare the environment, or `python bootstrap.py --self-test` to run a CPU prediction check.

Offline world boundaries come from UN Geospatial / OCHA; see `assets/maps/un_map_sources.json`. Map display does not imply UN endorsement of territorial status. US state boundaries come from Census TIGERweb, and the optional online road basemap is OpenStreetMap. Original third-party license notices are retained with the bundled resources.

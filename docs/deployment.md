# Streamlit deployment

Public URL: https://causal-workbench-t094.streamlit.app/

- Platform: Streamlit Community Cloud
- Repository: Mystic-commits/AML-DL_Project
- Branch: main
- Entrypoint: app.py
- Python: 3.12 (same version as local checks and CI)
- Dependencies: requirements.txt
- Secrets: none required

Classical inference reads bundled JSON snapshots. Neural inference downloads the checksum-verified public `phase2-v1.1` archive on first use and caches it by SHA-256. ONNX Runtime runs the portable unsigned-8-bit MatMul model on CPU with two threads. No model is trained at startup. Bayesian demo fitting is cached. Uploaded observations remain in the current Streamlit session and are not written to the repository or application filesystem by the app.

## Startup and usage

The free hosting instance may require a short wake-up after inactivity. Start with the guided example described in [app usage](usage.md). Text classification runs when its submit button is pressed. Intervention probabilities update when their controls change.

## Maintain or redeploy

Push reviewed changes to main. In the Streamlit workspace, use Manage app to inspect build logs, reboot, or change settings. For a fresh deployment, choose the coordinates above and set Python 3.12 under Advanced settings before deploying. Keep the local run command in README available as a fallback for internet interruptions.

## Visual system

Application palette: paper #F8F4EC, deep red #BF092F, warm ink #594832. Poppins fonts are locally bundled with their OFL license. The bundled grayscale research image is used on the overview page. App styling is in assets/workbench.css and .streamlit/config.toml.


## Verified neural release

The public asset is pinned in `artifacts/neural/release.json` (about 114.9 MB compressed, 138.9 MB ONNX model). Python runtime requirements include ONNX Runtime and tokenizers; the hosted app does not install the optional torch/transformers training stack. ZIP validation limits file names, paths and expanded sizes. CPU integration CI tests the actual downloaded model on Linux, including forward/reverse/no-relation examples. Quantized held-out edge F1 is 0.865762, compared with 0.870482 full precision; 16/2,700 argmax predictions differ. The superseded v1.0 signed-Gather model must not be used.

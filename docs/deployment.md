# Streamlit deployment

Public URL: https://causal-workbench-t094.streamlit.app/

- Platform: Streamlit Community Cloud
- Repository: Mystic-commits/AML-DL_Project
- Branch: main
- Entrypoint: app.py
- Python: 3.12 (same version as local checks and CI)
- Dependencies: requirements.txt
- Secrets: none required

The application reads bundled JSON model snapshots rather than downloading or training models at startup. Bayesian demo fitting is cached. Uploaded observations remain in the current Streamlit session and are not written to the repository or application filesystem by the app.

## Startup and usage

The free hosting instance may require a short wake-up after inactivity. Start with the guided example described in [app usage](usage.md). Text classification runs when its submit button is pressed. Intervention probabilities update when their controls change.

## Maintain or redeploy

Push reviewed changes to main. In the Streamlit workspace, use Manage app to inspect build logs, reboot, or change settings. For a fresh deployment, choose the coordinates above and set Python 3.12 under Advanced settings before deploying. Keep the local run command in README available as a fallback for internet interruptions.

## Visual system

Application palette: paper #F8F4EC, deep red #BF092F, warm ink #594832. Poppins fonts are locally bundled with their OFL license. The bundled grayscale research image is used on the overview page. App styling is in assets/workbench.css and .streamlit/config.toml.

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

## Before the viva

Open the public URL early so the free hosting instance is awake. Demonstrate Overview → Reasoning lab → Extraction benchmark → Text → DAG → Intervention lab. Predictions run only when their submit button is pressed. In Text → DAG, approve the desired edges before fitting a reviewed network; a proposed graph is not automatically a validated causal model.

## Maintain or redeploy

Push reviewed changes to main. In the Streamlit workspace, use Manage app to inspect build logs, reboot, or change settings. For a fresh deployment, choose the coordinates above and set Python 3.12 under Advanced settings before deploying. Keep the local run command in README available as a fallback for internet interruptions.

## Visual system

Presentation-matched palette: paper #F8F4EC, deep red #BF092F, warm ink #594832. Poppins fonts are locally bundled with their OFL license. Grayscale research imagery is shared with the presentation. App styling is in assets/workbench.css and .streamlit/config.toml.

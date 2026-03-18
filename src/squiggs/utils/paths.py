from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIGURES_DIR = PROJECT_ROOT / "figs"
FIGURES_DIR.mkdir(exist_ok=True)

# helper function to ensure the folder exists
def get_fig_path(custom_path: str | Path = None) -> Path:
    '''
    ensure the folder exists if a custom path, 
    otherwise just stick with the default path
    
    usage:
    fig_path = get_fig_path('/squigg_figs/my/custom/path/') # custom
    fig_path = get_fig_path()    # default, project_root/figs
    fig_path = paths.FIGURES_DIR # also default
    '''
    
    path = Path(custom_path) if custom_path else FIGURES_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path
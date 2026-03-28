# PyInstaller hook for flet
# Collects all flet modules and data files

from PyInstaller.utils.hooks import copy_metadata, collect_data_files, collect_submodules

# Collect all flet submodules
hiddenimports = collect_submodules('flet')
hiddenimports += collect_submodules('flet_core')
hiddenimports += collect_submodules('flet_desktop')
hiddenimports += collect_submodules('flet_runtime')

# Collect data files (icons, assets, etc.)
datas = copy_metadata('flet')
datas += copy_metadata('flet_core')
datas += copy_metadata('flet_desktop')
datas += copy_metadata('flet_runtime')

# Collect flet desktop binaries
try:
    datas += collect_data_files('flet_desktop')
except Exception:
    pass

from pathlib import Path

_deploy_root = Path(__file__).resolve().parent.parent
if str(_deploy_root) not in __path__:
    __path__.append(str(_deploy_root))

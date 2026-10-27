#!/usr/bin/env bash
set -euo pipefail

# Run inside Ubuntu 24.04 WSL as root (or through sudo).
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  mininet openvswitch-switch python3-venv curl tar

python3 -m venv /opt/cares-ryu
VENV=/opt/cares-ryu
$VENV/bin/pip install --upgrade pip setuptools wheel
$VENV/bin/pip install tinyrpc eventlet msgpack netaddr oslo.config ovs Routes WebOb pbr

# Ryu 4.34 predates Python 3.12 and assumes a setuptools hook removed upstream.
# Patch only the downloaded source for installation; the project controller stays standard Ryu.
mkdir -p /tmp/cares-ryu-build
curl -fsSL \
  https://files.pythonhosted.org/packages/25/30/1c1972b3f9bf52de32d2fdf7c5206ff7099e739bf69b6004f4a71bcbde18/ryu-4.34.tar.gz \
  -o /tmp/cares-ryu-build/ryu.tar.gz
tar -xzf /tmp/cares-ryu-build/ryu.tar.gz -C /tmp/cares-ryu-build
RYU_SRC=/tmp/cares-ryu-build/ryu-4.34
python3 - "$RYU_SRC/ryu/hooks.py" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
source = path.read_text()
source = source.replace(
    "_main_module()._orig_get_script_args = easy_install.get_script_args",
    '_main_module()._orig_get_script_args = getattr(easy_install, "get_script_args", easy_install.ScriptWriter.get_args)',
)
path.write_text(source)
PY
(cd "$RYU_SRC" && SETUPTOOLS_USE_DISTUTILS=local "$VENV/bin/python" setup.py install)
# Eventlet removed a websocket-only sentinel imported by Ryu's WSGI module.
# This baseline does not use Ryu's WebSocket API, but the import must still succeed.
python3 - "$VENV/lib/python3.12/site-packages/ryu/app/wsgi.py" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
source = path.read_text()
source = source.replace(
    '    from eventlet.wsgi import ALREADY_HANDLED\n    _ALREADY_HANDLED = ALREADY_HANDLED',
    '    try:\n        from eventlet.wsgi import ALREADY_HANDLED\n    except ImportError:\n        ALREADY_HANDLED = []\n    _ALREADY_HANDLED = ALREADY_HANDLED',
)
path.write_text(source)
PY

echo "Installed Mininet/OVS and Ryu 4.34 in $VENV"


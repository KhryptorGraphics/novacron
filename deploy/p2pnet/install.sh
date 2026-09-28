#!/usr/bin/env bash
set -Eeuo pipefail

(( EUID == 0 )) || { printf 'install.sh must run as root\n' >&2; exit 1; }
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if ! command -v rsync >/dev/null 2>&1; then
    apt-get install -y rsync
fi
rsync -a --delete --exclude lab --exclude tests "$source_dir/" /opt/p2pnet/
chown -R root:root /opt/p2pnet
find /opt/p2pnet -type d -exec chmod 0755 {} +
find /opt/p2pnet -type f -exec chmod 0644 {} +
find /opt/p2pnet -type f \( -name '*.sh' -o -name '*.py' -o -name 'mptcp-exec' \
    -o -name 'plane2-routes' -o -name 'ssh-p2p' -o -name 'zrepl-guard' \
    -o -path /opt/p2pnet/bin/p2pnet \) -exec chmod 0755 {} +
ln -sfn /opt/p2pnet/bin/p2pnet /usr/local/sbin/p2pnet
printf 'Installed p2pnet to /opt/p2pnet\n'

#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
P2PNET_SRC=$ROOT/deploy/p2pnet
LAB_DIR=${P2PNET_LAB_DIR:-$HOME/.cache/p2pnet-lab}
IMAGE_DIR=$LAB_DIR/images
RUN_DIR=$LAB_DIR/run
KEY_DIR=$LAB_DIR/keys
LOG=$RUN_DIR/test.log
ARCH=$(uname -m)
SSH_KEY=$KEY_DIR/lab_ed25519
SSH_OPTS=(-i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KEY_DIR/known_hosts" -o ConnectTimeout=5)

usage() { echo 'usage: lab.sh {up|provision|test [Tn...]|down [--purge]}' >&2; exit 2; }
require_host() {
    local cmd
    for cmd in qemu-img cloud-localds ssh ssh-keygen rsync curl sha256sum; do command -v "$cmd" >/dev/null || { echo "lab: missing host command: $cmd" >&2; exit 1; }; done
    [[ -r /dev/kvm && -w /dev/kvm ]] || { echo 'lab: /dev/kvm is unavailable to this user' >&2; exit 1; }
}
node_port() { echo $((2230 + ${1#node})); }
ssh_node() { local n=$1; shift; ssh "${SSH_OPTS[@]}" -p "$(node_port "$n")" "ubuntu@127.0.0.1" "$@"; }
scp_node() { local n=$1 source=$2 destination=$3; scp "${SSH_OPTS[@]}" -P "$(node_port "$n")" "$source" "ubuntu@127.0.0.1:$destination"; }
remote() { local n=$1 cmd=$2; ssh_node "$n" "sudo bash -lc $(printf '%q' "$cmd")"; }
pass() { printf 'PASS %s %s\n' "$1" "$2" | tee -a "$LOG"; }
fail() { printf 'FAIL %s %s\n' "$1" "$2" | tee -a "$LOG" >&2; exit 1; }
all_nodes() { printf '%s\n' node1 node2 node3; }
for_nodes() { local n; while IFS= read -r n; do "$1" "$n"; done < <(all_nodes); }
wait_ssh() {
    local n=$1 deadline=$((SECONDS + 900))
    until ssh_node "$n" 'cloud-init status --wait >/dev/null 2>&1 && true'; do
        (( SECONDS < deadline )) || return 1
        sleep 2
    done
}

fetch_verified() {
    local url=$1 destination=$2 sums_url=$3 filename digest
    filename=$(basename "$destination")
    mkdir -p "$(dirname "$destination")"
    if [[ ! -s $destination ]]; then curl -fL --retry 3 "$url" -o "$destination.tmp" && mv "$destination.tmp" "$destination"; fi
    curl -fL --retry 3 "$sums_url" -o "$destination.SHA256SUMS"
    digest=$(awk -v n="$filename" '$2 == n || $2 == "./" n || $2 == "*" n {print $1; exit}' "$destination.SHA256SUMS")
    [[ -n $digest ]] || { echo "lab: $filename absent from upstream SHA256SUMS" >&2; return 1; }
    printf '%s  %s\n' "$digest" "$destination" | sha256sum -c -
}

up() {
    require_host
    mkdir -p "$IMAGE_DIR" "$RUN_DIR" "$KEY_DIR"
    if [[ ! -f $SSH_KEY ]]; then ssh-keygen -q -t ed25519 -N '' -f "$SSH_KEY" -C p2pnet-lab; fi
    local cloud arch qemu machine cpu firmware vars cirros_base cirros_kernel cirros_initrd i user_data net_data disk pid block_device net_device
    if [[ $ARCH == aarch64 ]]; then
        cloud=noble-server-cloudimg-arm64.img; arch=aarch64; qemu='qemu-system-aarch64'; machine='virt,accel=kvm,gic-version=max'; cpu=host
        firmware=/usr/share/AAVMF/AAVMF_CODE.fd
        block_device=virtio-blk-device; net_device=virtio-net-device
        [[ -r $firmware && -r /usr/share/AAVMF/AAVMF_VARS.fd ]] || { echo 'lab: install AAVMF firmware files' >&2; return 1; }
        cirros_base=https://download.cirros-cloud.net/0.6.2/cirros-0.6.2-aarch64
    elif [[ $ARCH == x86_64 ]]; then
        cloud=noble-server-cloudimg-amd64.img; arch=x86_64; qemu='qemu-system-x86_64'; machine='q35,accel=kvm'; cpu=host
        firmware=/usr/share/OVMF/OVMF_CODE.fd
        block_device=virtio-blk-pci; net_device=virtio-net-pci
        [[ -r $firmware && -r /usr/share/OVMF/OVMF_VARS.fd ]] || { echo 'lab: install OVMF firmware files' >&2; return 1; }
        cirros_base=https://download.cirros-cloud.net/0.6.2/cirros-0.6.2-x86_64
    else echo "lab: unsupported host architecture $ARCH" >&2; return 1; fi
    command -v "$qemu" >/dev/null || { echo "lab: missing host command: $qemu" >&2; return 1; }
    fetch_verified "https://cloud-images.ubuntu.com/noble/current/$cloud" "$IMAGE_DIR/$cloud" https://cloud-images.ubuntu.com/noble/current/SHA256SUMS
    cirros_kernel=$IMAGE_DIR/cirros-kernel
    cirros_initrd=$IMAGE_DIR/cirros-initramfs
    for i in kernel initramfs; do
        if ! curl -fsSI "$cirros_base-$i" >/dev/null; then
            local archive=$IMAGE_DIR/cirros-0.6.2-$arch-uec.tar.gz
            curl -fL --retry 3 "$cirros_base-uec.tar.gz" -o "$archive"
            tar -tzf "$archive" >/dev/null
            tar -xzf "$archive" -C "$IMAGE_DIR"
            break
        fi
        curl -fL --retry 3 "$cirros_base-$i" -o "$IMAGE_DIR/cirros-$i"
    done
    if [[ ! -s $cirros_kernel ]]; then
        local candidate
        candidate=$(find "$IMAGE_DIR" -type f \( -iname '*kernel*' -o -iname '*vmlinuz*' \) ! -name 'cirros-kernel' ! -name '*.SHA256SUMS' -print -quit)
        [[ -n $candidate ]] && cp "$candidate" "$cirros_kernel"
    fi
    if [[ ! -s $cirros_initrd ]]; then
        local candidate
        candidate=$(find "$IMAGE_DIR" -type f \( -iname '*initramfs*' -o -iname '*initrd*' \) ! -name 'cirros-initramfs' ! -name '*.SHA256SUMS' -print -quit)
        [[ -n $candidate ]] && cp "$candidate" "$cirros_initrd"
    fi
    [[ -s $cirros_kernel && -s $cirros_initrd ]] || { echo 'lab: cirros kernel/initramfs unavailable' >&2; return 1; }
    for i in 1 2 3; do
        pid=$RUN_DIR/node$i.pid
        if [[ -f $pid ]] && kill -0 "$(<"$pid")" 2>/dev/null; then continue; fi
        disk=$RUN_DIR/node$i.qcow2
        if [[ ! -e $disk ]]; then qemu-img create -f qcow2 -F qcow2 -b "$IMAGE_DIR/$cloud" "$disk" >/dev/null; qemu-img resize "$disk" 20G >/dev/null; fi
        user_data=$(sed -e "s/__HOSTNAME__/node$i/g" -e "s|__SSH_PUBKEY__|$(cat "$SSH_KEY.pub")|g" "$P2PNET_SRC/lab/user-data.tmpl")
        net_data=$(sed "s/__INDEX__/$i/g" "$P2PNET_SRC/lab/network-config.tmpl")
        if (( i != 1 )); then net_data=$(sed '/^  wan2:/,$d' <<<"$net_data"); fi
        cloud-localds --network-config=<(printf '%s\n' "$net_data") "$RUN_DIR/node$i-seed.iso" <(printf '%s\n' "$user_data")
        vars=$RUN_DIR/node$i-vars.fd
        [[ -e $vars ]] || cp /usr/share/AAVMF/AAVMF_VARS.fd "$vars" 2>/dev/null || cp /usr/share/OVMF/OVMF_VARS.fd "$vars"
        local args=(-name "p2pnet-node$i" -machine "$machine" -cpu "$cpu" -smp 4 -m 4096 -display none -daemonize -pidfile "$pid" -serial "file:$RUN_DIR/node$i.serial.log" -drive "if=none,file=$disk,id=os,format=qcow2" -device "$block_device,drive=os" -drive "if=none,file=$RUN_DIR/node$i-seed.iso,id=seed,media=cdrom,readonly=on" -device "$block_device,drive=seed" -netdev "user,id=mgmt,hostfwd=tcp:127.0.0.1:$((2230+i))-:22" -device "$net_device,netdev=mgmt,mac=52:54:00:77:00:0$i" -netdev "socket,id=wan1,mcast=230.77.0.1:37701,localaddr=127.0.0.1" -device "$net_device,netdev=wan1,mac=52:54:00:77:01:0$i")
        args+=(-drive "if=pflash,format=raw,readonly=on,file=$firmware" -drive "if=pflash,format=raw,file=$vars")
        if (( i == 1 )); then args+=(-netdev "socket,id=wan2,mcast=230.77.0.1:37701,localaddr=127.0.0.1" -device "$net_device,netdev=wan2,mac=52:54:00:77:02:01"); fi
        "$qemu" "${args[@]}"
    done
    ssh-keyscan -p 2231 127.0.0.1 >"$KEY_DIR/known_hosts" 2>/dev/null || true
    local n
    for n in node1 node2 node3; do wait_ssh "$n" || { echo "lab: cloud-init/SSH did not become ready on $n; see $RUN_DIR/$n.serial.log" >&2; return 1; }; done
}

provision() {
    local n
    for n in node1 node2 node3; do
        rsync -a --delete --exclude lab --exclude tests "$P2PNET_SRC/" -e "ssh -i $SSH_KEY -p $(node_port "$n") -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KEY_DIR/known_hosts" "ubuntu@127.0.0.1:/tmp/p2pnet-src/"
        remote "$n" "apt-get update && apt-get install -y rsync && /tmp/p2pnet-src/install.sh && p2pnet deps install && apt-get install -y linux-modules-extra-\$(uname -r) && p2pnet init --node $n"
        scp_node "$n" "$IMAGE_DIR/cirros-kernel" /tmp/cirros-kernel
        scp_node "$n" "$IMAGE_DIR/cirros-initramfs" /tmp/cirros-initramfs
        remote "$n" 'install -m 0644 /tmp/cirros-kernel /var/lib/libvirt/images/cirros-kernel && install -m 0644 /tmp/cirros-initramfs /var/lib/libvirt/images/cirros-initramfs && p2pnet keys' >"$KEY_DIR/$n.keys"
        ssh_node "$n" 'sudo cat /etc/wireguard/wg0.pub' >"$KEY_DIR/$n.wg0.pub"
        ssh_node "$n" 'sudo cat /etc/p2pnet/ssh/id_ed25519.pub' >"$KEY_DIR/$n.ssh.pub"
        ssh_node "$n" 'sudo cat /var/lib/p2pnet/bulk/.ssh/id_ed25519.pub' >"$KEY_DIR/$n.bulk.pub"
        ssh_node "$n" 'sudo cat /etc/ssh/ssh_host_ed25519_key.pub' >"$KEY_DIR/$n.host.pub"
        if [[ $n == node1 ]]; then
            remote "$n" 'p2pnet keys --wan2 --restic-password'
            ssh_node "$n" 'sudo cat /etc/wireguard/wg1.pub' >"$KEY_DIR/$n.wg1.pub"
        fi
    done
    local n
    for n in node1 node2 node3; do
        mkdir -p "$KEY_DIR/$n"
        cp "$KEY_DIR/$n.wg0.pub" "$KEY_DIR/$n/wg0.pub"
        cp "$KEY_DIR/$n.ssh.pub" "$KEY_DIR/$n/ssh.pub"
        cp "$KEY_DIR/$n.bulk.pub" "$KEY_DIR/$n/bulk.pub"
        cp "$KEY_DIR/$n.host.pub" "$KEY_DIR/$n/host.pub"
        [[ ! -f $KEY_DIR/$n.wg1.pub ]] || cp "$KEY_DIR/$n.wg1.pub" "$KEY_DIR/$n/wg1.pub"
    done
    /usr/bin/python3 "$P2PNET_SRC/lab/lab_inventory.py" --keys-dir "$KEY_DIR" --output "$RUN_DIR/inventory.yaml"
    for n in node1 node2 node3; do
        scp_node "$n" "$RUN_DIR/inventory.yaml" /tmp/inventory.yaml
        remote "$n" 'install -o root -g root -m 0644 /tmp/inventory.yaml /etc/p2pnet/inventory.yaml'
    done
    ssh_node node1 'sudo cat /etc/p2pnet/secrets/restic.pass' >"$KEY_DIR/restic.pass"
    for n in node2 node3; do scp_node "$n" "$KEY_DIR/restic.pass" /tmp/restic.pass; remote "$n" 'install -o root -g p2pbulk -m 0640 /tmp/restic.pass /etc/p2pnet/secrets/restic.pass'; done
    for n in node1 node2 node3; do
        remote "$n" 'zpool list tank >/dev/null 2>&1 || { truncate -s 6G /var/lib/p2pnet-lab-zpool.img && zpool create -f tank /var/lib/p2pnet-lab-zpool.img; }'
        remote "$n" 'ip link show wan1 >/dev/null; tc qdisc replace dev wan1 handle ffff: ingress; modprobe ifb; ip link add ifb0 type ifb 2>/dev/null || true; ip link set ifb0 up; tc filter replace dev wan1 parent ffff: protocol all u32 match u32 0 0 action mirred egress redirect dev ifb0; tc qdisc replace dev ifb0 root netem delay 20ms'
        if [[ $n == node1 ]]; then
            remote "$n" 'tc qdisc replace dev wan2 handle ffff: ingress; ip link add ifb1 type ifb 2>/dev/null || true; ip link set ifb1 up; tc filter replace dev wan2 parent ffff: protocol all u32 match u32 0 0 action mirred egress redirect dev ifb1; tc qdisc replace dev ifb1 root netem delay 20ms'
        fi
    done
    remote node1 'zfs list tank/vms >/dev/null 2>&1 || { zfs create tank/vms && zfs create -V 256M tank/vms/vm1 && dd if=/dev/urandom of=/dev/zvol/tank/vms/vm1 bs=1M count=128 status=none && zfs create tank/vms/files && dd if=/dev/urandom of=/tank/vms/files/blob bs=1M count=64 status=none; }'
}

install_all() { local n c; for n in node1 node2 node3; do for c in "$@"; do remote "$n" "p2pnet $c install"; done; done; }
verify_all() { local n c; for n in node1 node2 node3; do for c in "$@"; do remote "$n" "p2pnet $c verify"; done; done; }

T1() { local n; for n in node1 node2 node3; do remote "$n" 'p2pnet inventory validate'; done; }
T2() { install_all wg && verify_all wg; }
T3() {
    install_all l2 && verify_all l2 || return 1
    (remote node1 'p2pnet l2 selftest --peer node2' >"$RUN_DIR/l2-node1.log" 2>&1) & local a=$!
    (remote node2 'p2pnet l2 selftest --peer node1' >"$RUN_DIR/l2-node2.log" 2>&1) & local b=$!
    wait "$a" && wait "$b" && verify_all l2
}
T4() { install_all tune && verify_all tune; }
T5() {
    install_all bench
    local out
    out=$(remote node1 'p2pnet bench run node2 --json') || return 1
    printf '%s\n' "$out" | tee -a "$LOG"
    /usr/bin/python3 -c 'import json,sys; x=json.loads(sys.stdin.read().splitlines()[-1]); assert x.get("verdict"); assert float(x["overlay_mbit"]["eight_streams"]) >= 100' <<<"$out"
}
T6() {
    install_all mptcp && verify_all mptcp
    remote node1 'p2pnet mptcp verify --peer node2'
    remote node1 'tc qdisc replace dev wan1 root tbf rate 50mbit burst 64k latency 50ms; tc qdisc replace dev wan2 root tbf rate 50mbit burst 64k latency 50ms'
    local multi single
    multi=$(remote node1 'p2pnet mptcp-run -- iperf3 -c 10.77.0.2 -t 10 -J') || { remote node1 'p2pnet tune apply-qdisc'; return 1; }
    single=$(remote node1 'iperf3 -c 10.77.0.2 -t 10 -J') || { remote node1 'p2pnet tune apply-qdisc'; return 1; }
    remote node1 'p2pnet tune apply-qdisc'
    /usr/bin/python3 -c 'import json,sys; a=json.loads(sys.argv[1])["end"]["sum_received"]["bits_per_second"]; b=json.loads(sys.argv[2])["end"]["sum_received"]["bits_per_second"]; assert a >= 1.5*b, (a,b)' "$multi" "$single"
}
T7() {
    install_all sshd && verify_all sshd || return 1
    if remote node1 'ssh -i /etc/p2pnet/ssh/id_ed25519 -o StrictHostKeyChecking=no -p 2222 p2prepl@10.77.0.2 id >/dev/null 2>&1'; then
        echo 'restricted replication key unexpectedly accepted id' >&2
        return 1
    fi
}
T8() { install_all qos && verify_all qos && remote node1 'p2pnet qos selftest --peer node2'; }
class_bytes() {
    local node=$1 classid=$2
    remote "$node" "tc -s class show dev wg0 classid $classid | awk '/Sent/ {print \$2; exit}'"
}
T9() {
    install_all libvirt && verify_all libvirt
    local arch machine cpu console emulator before after n
    if [[ $ARCH == aarch64 ]]; then arch=aarch64; machine=virt; cpu="<cpu mode='maximum' check='none'/>"; console=ttyAMA0; emulator=/usr/bin/qemu-system-aarch64; else arch=x86_64; machine=q35; cpu="<cpu mode='custom'><model>qemu64</model></cpu>"; console=ttyS0; emulator=/usr/bin/qemu-system-x86_64; fi
    for n in node1 node2; do
        remote "$n" 'if virsh dominfo cirros >/dev/null 2>&1; then virsh destroy cirros >/dev/null 2>&1 || true; virsh undefine cirros; fi; rm -f /var/lib/libvirt/images/cirros-data.qcow2'
    done
    remote node1 'qemu-img create -f qcow2 /var/lib/libvirt/images/cirros-data.qcow2 64M'
    remote node1 'qemu-io -f qcow2 -c "write -P 0x5a 0 4096" /var/lib/libvirt/images/cirros-data.qcow2'
    remote node2 'qemu-img create -f qcow2 /var/lib/libvirt/images/cirros-data.qcow2 64M'
    local xml
    xml=$(sed -e "s/__ARCH__/$arch/g" -e "s/__MACHINE__/$machine/g" -e "s|__CPU_XML__|$cpu|g" -e "s/__CONSOLE__/$console/g" -e "s|__EMULATOR__|$emulator|g" "$P2PNET_SRC/lab/cirros-domain.xml.tmpl")
    printf '%s\n' "$xml" >"$RUN_DIR/cirros.xml"
    scp_node node1 "$RUN_DIR/cirros.xml" /tmp/cirros.xml
    remote node1 'virsh define /tmp/cirros.xml && virsh start cirros'
    remote node1 'p2pnet migrate plan cirros node2'
    before=$(class_bytes node1 1:20)
    remote node1 'p2pnet migrate run cirros node2 --profile multifd --copy-storage'
    remote node2 'virsh domstate cirros | grep -qi running'
    remote node1 'test -s /var/lib/p2pnet/migrate/last.json && jq -e .downtime_ms /var/lib/p2pnet/migrate/last.json >/dev/null'
    after=$(class_bytes node1 1:20)
    (( after - before >= 1048576 ))
    remote node2 'virsh destroy cirros >/dev/null && qemu-io -r -f qcow2 -c "read -P 0x5a 0 4096" /var/lib/libvirt/images/cirros-data.qcow2'
}
T10() {
    install_all zrepl
    remote node1 'p2pnet zrepl run labjob && p2pnet zrepl run labjob'
    remote node1 'dd if=/dev/urandom of=/dev/zvol/tank/vms/vm1 bs=1M count=200 conv=notrunc status=none'
    local log_offset target
    log_offset=$(remote node1 'wc -c < /var/log/p2pnet/zrepl.log')
    remote node1 'p2pnet zrepl run labjob' >"$RUN_DIR/zrepl-interrupt.log" 2>&1 & local sender=$!
    # zfs send drains fast into the 1 GiB mbuffer; interrupt the WAN leg instead.
    # shellcheck disable=SC2016
    ssh_node node1 'for i in $(seq 240); do pgrep -x mbuffer >/dev/null && break; sleep 0.25; done; sleep 2'
    ssh_node node1 'sudo pkill -x mbuffer || true'
    ssh_node node2 'sudo pkill -x mbuffer || true'
    wait "$sender" || true
    remote node1 'p2pnet zrepl run labjob'
    remote node1 "tail -c +$((log_offset + 1)) /var/log/p2pnet/zrepl.log | grep -q 'action=resume'"
    remote node1 'p2pnet zrepl run labjob'
    for target in node2 node3; do
# shellcheck disable=SC2016
        remote "$target" 'for ds in $(zfs list -H -o name -r -t filesystem,volume tank/p2p-replicas/node1/vms); do n=$(zfs list -H -t snapshot -o name "$ds" | grep -c @p2pnet- || true); test "$n" -le 4 || exit 1; done'
    done
}
T11() {
    install_all swarm
    for n in node2 node3; do remote "$n" 'systemctl stop p2pnet-swarmd.service && rm -f /var/lib/p2pnet/images/lab-base.img /var/lib/p2pnet/swarm/queue/*.torrent /var/lib/p2pnet/swarm/resume/*.fastresume && systemctl start p2pnet-swarmd.service' || return 1; done
    local before after hashes n
    before=$(class_bytes node1 1:40)
    remote node1 'dd if=/dev/urandom of=/var/lib/p2pnet/images/lab-base.img bs=1M count=1024 status=none && p2pnet swarm publish /var/lib/p2pnet/images/lab-base.img'
    (remote node2 'p2pnet swarm fetch lab-base.img --wait --timeout 900') & local a=$!
    (remote node3 'p2pnet swarm fetch lab-base.img --wait --timeout 900') & local b=$!
    wait "$a" && wait "$b"
    hashes=$(for n in node1 node2 node3; do remote "$n" 'sha256sum /var/lib/p2pnet/images/lab-base.img' || return 1; done | awk '{print $1}' | sort -u | wc -l)
    [[ $hashes == 1 ]] || return 1
    after=$(class_bytes node1 1:40)
    (( after - before >= 536870912 ))
}
T12() {
    install_all dedup || return 1
    remote node1 'p2pnet dedup init --seed'
    remote node2 'p2pnet dedup init --from node1'
    remote node1 'dd if=/dev/urandom of=/tmp/base.img bs=1M count=512 status=none && p2pnet dedup ingest /tmp/base.img --name base'
    remote node2 'p2pnet dedup fetch base --from node1 --output /tmp/base.img'
    remote node1 'cp /tmp/base.img /tmp/derived.img && dd if=/dev/urandom of=/tmp/patch bs=1M count=16 status=none && dd if=/tmp/patch of=/tmp/derived.img bs=1M seek=256 conv=notrunc status=none && p2pnet dedup ingest /tmp/derived.img --name derived'
    remote node2 'p2pnet dedup fetch derived --from node1 --output /tmp/derived.img'
    remote node2 'jq -e ".transferred_bytes <= 67108864 and .savings_pct >= 85" /var/lib/p2pnet/dedup/last-fetch.json'
}
T13() { remote node1 'p2pnet place --all --json' | /usr/bin/python3 -c 'import json,sys; x=json.load(sys.stdin); x=x[0] if isinstance(x,list) else x; assert x["node"]=="node1" and x["action"]=="stay" and x["wan_copy_gib"]==0'; }
T14() { install_all schedule && verify_all schedule && for_nodes health_node; }
health_node() { remote "$1" 'p2pnet health'; }
T15() {
    ssh_node node3 'sudo cp /var/lib/p2pnet/state/tune.prev /tmp/p2pnet-tune.before'
    remote node3 'p2pnet all uninstall'
    remote node3 'test ! -e /sys/class/net/wg0 && ! ip link show vxlan0 >/dev/null 2>&1 && ! ip link show br-p2p >/dev/null 2>&1 && ! tc qdisc show dev wg0 | grep -q htb && ! systemctl list-unit-files | grep -q p2pnet'
# shellcheck disable=SC2016
    remote node3 'while IFS="=" read -r key value; do test "$(sysctl -n "$key")" = "$value" || exit 1; done </tmp/p2pnet-tune.before'
    remote node3 'p2pnet tune install && p2pnet wg install && p2pnet l2 install && p2pnet mptcp install && p2pnet bench install && p2pnet sshd install && p2pnet qos install && p2pnet libvirt install && p2pnet zrepl install && p2pnet swarm install && p2pnet dedup install && p2pnet dedup init --from node1 && p2pnet schedule install && p2pnet all verify'
}
T16() {
    local c output
    for c in tune wg l2 mptcp bench sshd qos libvirt zrepl swarm dedup schedule; do
        output=$(remote node2 "p2pnet $c install") || return 1
        grep -q "p2pnet\[$c\] install complete: changed=0" <<<"$output" || { echo "$c not idempotent: $output" >&2; return 1; }
    done
}

test_lab() {
    mkdir -p "$RUN_DIR"; : >"$LOG"
    local requested id description fn rc
    local -a requested_ids
    if (($#)); then requested=$(printf '%s\n' "$@"); else requested=$(printf 'T%s\n' {1..16}); fi
    mapfile -t requested_ids <<<"$requested"
    local -A cases=(
        [T1]='inventory validate' [T2]='WireGuard mesh and handshake' [T3]='VXLAN bridge and cross-node L2'
        [T4]='high-BDP kernel tuning' [T5]='overlay bottleneck benchmark' [T6]='dual-WAN MPTCP aggregation'
        [T7]='restricted overlay SSH' [T8]='HTB classes and traffic classification'
        [T9]='libvirt live migration and QoS traffic' [T10]='incremental, resumable ZFS replication'
        [T11]='multi-source swarm image distribution' [T12]='content-defined deduplication'
        [T13]='VM placement' [T14]='off-peak timers and full health check'
        [T15]='uninstall and reinstallation' [T16]='component idempotency'
    )
    for id in "${requested_ids[@]}"; do
        [[ -n $id ]] || continue
        [[ -v cases[$id] ]] || fail "$id" 'unknown lab test'
        description=${cases[$id]}; fn=$id
        set +e; ( set -e; "$fn" ); rc=$?; set -e
        if (( rc == 0 )); then pass "$id" "$description"; else fail "$id" "$description"; fi
    done
}

down() {
    local purge=0 n pid deadline
    if (($# > 1)); then usage; fi
    if (($# == 1)); then [[ $1 == --purge ]] || usage; fi
    [[ ${1:-} != --purge ]] || purge=1
    for n in node1 node2 node3; do
        pid=$RUN_DIR/$n.pid
        if [[ -f $pid ]]; then
            pid=$(<"$pid")
            kill "$pid" 2>/dev/null || true
            deadline=$((SECONDS + 10))
            while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 0.2; done
            if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; fi
            rm -f "$RUN_DIR/$n.pid"
        fi
    done
    rm -rf "$RUN_DIR"
    if (( purge )); then rm -rf "$IMAGE_DIR"; fi
}

case "${1:-}" in
    up) shift; up "$@" ;;
    provision) shift; provision "$@" ;;
    test) shift; test_lab "$@" ;;
    down) shift; down "$@" ;;
    *) usage ;;
esac

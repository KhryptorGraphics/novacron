#!/usr/bin/env bats

setup() {
    export P2PNET_COMMON="$BATS_TEST_DIRNAME/../../libexec/common.sh"
    export P2PNET_NODE_ENV="$BATS_TEST_DIRNAME/fixtures/node.env"
    export DRY_RUN=1
    export PATH="$BATS_TEST_TMPDIR/bin:$PATH"
    mkdir -p "$BATS_TEST_TMPDIR/bin"
    cat > "$BATS_TEST_TMPDIR/bin/zfs" <<'EOF'
#!/usr/bin/env bash
case "$1" in
    get) printf -- '-\n' ;;
    list)
        if [[ " $* " == *" -t snapshot "* ]]; then
            printf 'tank/p2p-replicas/node-b/vms@p2pnet-20260927T010000Z\t1\n'
        else
            printf 'tank/p2p-replicas/node-b/vms\n'
        fi
        ;;
esac
EOF
    chmod +x "$BATS_TEST_TMPDIR/bin/zfs"
    export ZREPL_GUARD="$BATS_TEST_DIRNAME/../../libexec/zrepl-guard"
}

run_guard() {
    run env SSH_ORIGINAL_COMMAND="$1" "$ZREPL_GUARD" node-b
}

@test "allows state for a receiver path beneath this peer" {
    run_guard "state tank/p2p-replicas/node-b/vms"
    [ "$status" -eq 0 ]
    [[ "$output" == *"token=-"* ]]
    [[ "$output" == *"latest=p2pnet-20260927T010000Z"* ]]
}

@test "allows dry-run receive, abort and prune for a receiver path" {
    run_guard "recv tank/p2p-replicas/node-b/vms"
    [ "$status" -eq 0 ]
    [[ "$output" == *"zfs receive -s -u"* ]]

    run_guard "abort tank/p2p-replicas/node-b/vms"
    [ "$status" -eq 0 ]
    [[ "$output" == *"zfs receive -A"* ]]

    run_guard "prune tank/p2p-replicas/node-b/vms 3 2"
    [ "$status" -eq 0 ]
    [[ "$output" == *"zfs list"* ]]
}

@test "denies unrecognized commands and arguments" {
    run_guard "id"
    [ "$status" -ne 0 ]
    run_guard "state tank/p2p-replicas/node-b/vms extra"
    [ "$status" -ne 0 ]
}

@test "denies paths outside the peer subtree, traversal and snapshot syntax" {
    run_guard "state tank/p2p-replicas/node-c/vms"
    [ "$status" -ne 0 ]
    run_guard "state tank/p2p-replicas/node-b/../outside"
    [ "$status" -ne 0 ]
    run_guard "state tank/p2p-replicas/node-b/vms@p2pnet-20260927T010000Z"
    [ "$status" -ne 0 ]
}

@test "denies non-integer retention counts" {
    run_guard "prune tank/p2p-replicas/node-b/vms nope 2"
    [ "$status" -ne 0 ]
    run_guard "prune tank/p2p-replicas/node-b/vms 3 nope"
    [ "$status" -ne 0 ]
}

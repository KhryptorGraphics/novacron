#!/usr/bin/env bats

setup() {
  export P2PNET_NODE_ENV="$BATS_TEST_DIRNAME/fixtures/node.env"
  export P2PNET_MPTCP_SYSCTL=/nonexistent
}

@test "falls back to plain TCP with an explanatory message when MPTCP is unavailable" {
  run "$BATS_TEST_DIRNAME/../../libexec/mptcp-exec" /bin/echo ok
  [ "$status" -eq 0 ]
  [ "$output" = $'p2pnet: MPTCP unavailable (kernel MPTCP sysctl /nonexistent is unavailable); running over plain TCP\nok' ]
}

@test "requires a command" {
  run "$BATS_TEST_DIRNAME/../../libexec/mptcp-exec"
  [ "$status" -eq 2 ]
}

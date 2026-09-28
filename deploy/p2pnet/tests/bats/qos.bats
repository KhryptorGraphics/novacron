#!/usr/bin/env bats

setup() {
  export P2PNET_NODE_ENV="$BATS_TEST_DIRNAME/fixtures/node.env"
}

@test "dry-run builds the expected HTB rates, priorities, and default class" {
  output=$(DRY_RUN=1 "$BATS_TEST_DIRNAME/../../libexec/qos.sh" apply --profile night --dry-run 2>&1)
  [[ "$output" == *"qdisc replace dev wg0 root handle 1: htb default 30"* ]]
  [[ "$output" == *"classid 1:1 htb rate 902400kbit ceil 902400kbit"* ]]
  [[ "$output" == *"classid 1:10 htb rate 9600kbit ceil 902400kbit prio 0"* ]]
  [[ "$output" == *"classid 1:20 htb rate 192000kbit ceil 902400kbit prio 1"* ]]
  [[ "$output" == *"classid 1:25 htb rate 96000kbit ceil 902400kbit prio 2"* ]]
  [[ "$output" == *"classid 1:30 htb rate 595776kbit ceil 902400kbit prio 3"* ]]
  [[ "$output" == *"classid 1:40 htb rate 9024kbit ceil 483840kbit prio 4"* ]]
  [[ "$output" == *"qdisc replace dev wg0 parent 1:40 handle 40: fq_codel"* ]]
}

@test "dry-run renders migration, replication, bulk, and control classifiers" {
  output=$(DRY_RUN=1 "$BATS_TEST_DIRNAME/../../libexec/qos.sh" apply --profile day --dry-run 2>&1)
  [[ "$output" == *"match ip dport 49152 0xffc0 flowid 1:20"* ]]
  [[ "$output" == *"match ip dport 2222 0xffff flowid 1:25"* ]]
  [[ "$output" == *"match ip dport 6881 0xffff flowid 1:40"* ]]
  [[ "$output" == *"match ip sport 6881 0xffff flowid 1:40"* ]]
  [[ "$output" == *"match ip dport 2223 0xffff flowid 1:40"* ]]
  [[ "$output" == *"match ip sport 2223 0xffff flowid 1:40"* ]]
  [[ "$output" == *"match ip dport 22 0xffff match u16 0x0000 0xfe00 at 2 flowid 1:10"* ]]
  [[ "$output" == *"ceil 90240kbit prio 4"* ]]
}

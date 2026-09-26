package policy

import (
	"testing"
	"time"

	"github.com/sirupsen/logrus"
)

func newTestEvaluator() *DefaultPolicyEvaluator {
	return NewDefaultPolicyEvaluator(logrus.New())
}

func TestScheduleActiveAtCronMinute(t *testing.T) {
	// 2026-01-05 is a Monday. 09:00 UTC matches "0 9 * * 1".
	mondayNine := time.Date(2026, 1, 5, 9, 0, 30, 0, time.UTC)
	mondayOther := time.Date(2026, 1, 5, 10, 0, 0, 0, time.UTC)

	e := newTestEvaluator()
	schedule := &RuleSchedule{
		Enabled:        true,
		CronExpression: "0 9 * * 1",
	}
	if !e.scheduleActiveAt(schedule, mondayNine) {
		t.Fatal("monday 09:00 was inactive")
	}
	if e.scheduleActiveAt(schedule, mondayOther) {
		t.Fatal("monday 10:00 was active")
	}
}

func TestScheduleInactiveForInvalidCron(t *testing.T) {
	e := newTestEvaluator()
	schedule := &RuleSchedule{
		Enabled:        true,
		CronExpression: "not a cron",
	}
	if e.scheduleActiveAt(schedule, time.Date(2026, 1, 5, 9, 0, 0, 0, time.UTC)) {
		t.Fatal("invalid cron was treated as active")
	}
}

func TestScheduleRespectsTimezone(t *testing.T) {
	// 14:00 UTC is 09:00 in America/New_York during standard time.
	instant := time.Date(2026, 1, 5, 14, 0, 0, 0, time.UTC)
	e := newTestEvaluator()
	schedule := &RuleSchedule{
		Enabled:        true,
		CronExpression: "0 9 * * 1",
		Timezone:       "America/New_York",
	}
	if !e.scheduleActiveAt(schedule, instant) {
		t.Fatal("timezone-adjusted minute did not match")
	}
}

func TestScheduleBoundsStillGateCron(t *testing.T) {
	e := newTestEvaluator()
	start := time.Date(2026, 1, 5, 12, 0, 0, 0, time.UTC)
	schedule := &RuleSchedule{
		Enabled:        true,
		CronExpression: "* * * * *",
		StartTime:      &start,
	}
	before := time.Date(2026, 1, 5, 9, 0, 0, 0, time.UTC)
	if e.scheduleActiveAt(schedule, before) {
		t.Fatal("cron matched before the schedule start")
	}
}

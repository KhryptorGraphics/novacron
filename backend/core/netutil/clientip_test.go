package netutil

import (
	"net/http"
	"net/http/httptest"
	"net/netip"
	"testing"
)

func TestParseTrustedProxies(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want []string
	}{
		{"empty", "", nil},
		{"bare ip", "127.0.0.1", []string{"127.0.0.1/32"}},
		{"cidr masked", "10.1.2.3/8", []string{"10.0.0.0/8"}},
		{"mixed with whitespace", " 127.0.0.1 , 10.0.0.0/8 ,::1 ", []string{"127.0.0.1/32", "10.0.0.0/8", "::1/128"}},
		{"invalid entry dropped", "127.0.0.1, not-an-ip, 10.0.0.0/8", []string{"127.0.0.1/32", "10.0.0.0/8"}},
		{"all invalid", "garbage,,more garbage", nil},
		{"ipv4-mapped bare ip unmapped", "::ffff:10.0.0.5", []string{"10.0.0.5/32"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ParseTrustedProxies(tc.raw)
			if len(got) != len(tc.want) {
				t.Fatalf("ParseTrustedProxies(%q) = %v, want %v", tc.raw, got, tc.want)
			}
			for i := range got {
				if got[i].String() != tc.want[i] {
					t.Fatalf("ParseTrustedProxies(%q)[%d] = %s, want %s", tc.raw, i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestClientIP(t *testing.T) {
	cases := []struct {
		name       string
		remoteAddr string
		xff        []string // one entry per X-Forwarded-For header line
		xRealIP    string
		trusted    string
		want       string
	}{
		{"untrusted peer: XFF spoof ignored", "203.0.113.11:34567", []string{"198.51.100.1"}, "", "", "203.0.113.11"},
		{"trusted peer: single-hop XFF honored", "127.0.0.1:44321", []string{"198.51.100.7"}, "", "127.0.0.1", "198.51.100.7"},
		{"trusted peer: rightmost trusted hop skipped, one line", "127.0.0.1:44321", []string{"198.51.100.7, 127.0.0.1"}, "", "127.0.0.1, 10.0.0.0/8", "198.51.100.7"},
		{"trusted peer: rightmost trusted hop skipped, two header lines", "127.0.0.1:44321", []string{"198.51.100.7", "127.0.0.1"}, "", "127.0.0.1, 10.0.0.0/8", "198.51.100.7"},
		{"client-forged entries left of the boundary ignored", "127.0.0.1:44321", []string{"6.6.6.6, 198.51.100.7"}, "", "127.0.0.1", "198.51.100.7"},
		{"client-forged entries left of the boundary ignored, two lines", "127.0.0.1:44321", []string{"6.6.6.6", "198.51.100.7"}, "", "127.0.0.1", "198.51.100.7"},
		{"unparsable nearest hop is the boundary: never skipped to a forged entry", "127.0.0.1:44321", []string{"6.6.6.6, unknown"}, "", "127.0.0.1", "127.0.0.1"},
		{"multi-hop with intermediate trusted proxy", "10.1.2.3:1", []string{"203.0.113.5, 10.9.9.9, 10.1.2.3"}, "", "10.0.0.0/8", "203.0.113.5"},
		{"ipv4-mapped trusted hop is normalized then skipped", "10.1.2.3:1", []string{"203.0.113.5, ::ffff:10.0.0.5"}, "", "10.0.0.0/8", "203.0.113.5"},
		{"all hops trusted: peer returned", "10.1.2.3:1", []string{"10.9.9.9, 10.8.8.8"}, "", "10.0.0.0/8", "10.1.2.3"},
		{"untrusted peer outside CIDR: headers ignored", "203.0.113.12:34567", []string{"198.51.100.9"}, "", "127.0.0.1, 10.0.0.0/8", "203.0.113.12"},
		{"trusted peer via CIDR honors XFF", "10.96.0.2:34567", []string{"198.51.100.11"}, "", "10.0.0.0/8", "198.51.100.11"},
		{"garbage-only XFF from trusted peer: peer returned", "127.0.0.1:1", []string{"not-an-ip"}, "", "127.0.0.1", "127.0.0.1"},
		{"empty XFF line from trusted peer: X-Real-IP fallback", "127.0.0.1:1", []string{"  "}, "198.51.100.20", "127.0.0.1", "198.51.100.20"},
		{"garbage RemoteAddr: no address", "garbage-value", nil, "", "", ""},
		{"portless RemoteAddr, IPv4", "203.0.113.9", nil, "", "", "203.0.113.9"},
		{"portless RemoteAddr, IPv6", "2001:db8::1", nil, "", "", "2001:db8::1"},
		{"bracketed IPv6 peer + port, trusted XFF", "[::1]:8080", []string{"2001:db8::5"}, "", "::1", "2001:db8::5"},
		{"zoned IPv6 peer normalized", "[fe80::1%eth0]:9", []string{"198.51.100.1"}, "", "", "fe80::1"},
		{"zoned IPv6 XFF hop normalized", "127.0.0.1:1", []string{"fe80::2%eth1"}, "", "127.0.0.1", "fe80::2"},
		{"ipv4-mapped peer normalized and trusted", "[::ffff:127.0.0.1]:5", []string{"198.51.100.3"}, "", "127.0.0.1", "198.51.100.3"},
		{"X-Real-IP fallback when XFF absent, trusted peer", "127.0.0.1:1", nil, "198.51.100.20", "127.0.0.1", "198.51.100.20"},
		{"X-Real-IP ignored from untrusted peer", "203.0.113.1:1", nil, "198.51.100.20", "", "203.0.113.1"},
		{"X-Real-IP garbage from trusted peer: peer returned", "127.0.0.1:1", nil, "garbage", "127.0.0.1", "127.0.0.1"},
		{"XFF present wins over X-Real-IP", "127.0.0.1:1", []string{"198.51.100.7"}, "198.51.100.20", "127.0.0.1", "198.51.100.7"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/", nil)
			req.RemoteAddr = tc.remoteAddr
			for _, line := range tc.xff {
				req.Header.Add("X-Forwarded-For", line)
			}
			if tc.xRealIP != "" {
				req.Header.Set("X-Real-IP", tc.xRealIP)
			}
			trusted := ParseTrustedProxies(tc.trusted)
			got := ClientIPString(req, trusted)
			if got != tc.want {
				t.Fatalf("ClientIPString = %q, want %q", got, tc.want)
			}
			addr := ClientIP(req, trusted)
			if tc.want == "" {
				if addr.IsValid() {
					t.Fatalf("ClientIP = %v, want zero Addr", addr)
				}
				return
			}
			if !addr.IsValid() || addr.Zone() != "" || addr.Is4In6() {
				t.Fatalf("ClientIP = %v, want a valid, unmapped, zone-free address", addr)
			}
		})
	}
}

func TestClientIPNilRequest(t *testing.T) {
	if got := ClientIP(nil, nil); got != (netip.Addr{}) {
		t.Fatalf("ClientIP(nil) = %v, want zero Addr", got)
	}
	if got := ClientIPString(nil, nil); got != "" {
		t.Fatalf("ClientIPString(nil) = %q, want empty", got)
	}
}

func TestHostOf(t *testing.T) {
	cases := map[string]string{
		"203.0.113.9:1234":  "203.0.113.9",
		"[2001:db8::1]:443": "2001:db8::1",
		"[fe80::1%eth0]:9":  "fe80::1%eth0",
		"203.0.113.9":       "203.0.113.9",
		"2001:db8::1":       "2001:db8::1",
		" garbage ":         "garbage",
	}
	for in, want := range cases {
		if got := HostOf(in); got != want {
			t.Fatalf("HostOf(%q) = %q, want %q", in, got, want)
		}
	}
}

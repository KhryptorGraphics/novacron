// Package netutil holds the one shared client-IP resolution and trusted-proxy
// parsing implementation used by every NovaCron HTTP entry point that has to
// decide whether to believe proxy-supplied headers (api-server's login rate
// limiter, the hypervisor binary's session store).
package netutil

import (
	"net"
	"net/http"
	"net/netip"
	"strings"
)

// ParseTrustedProxies parses a comma-separated list of IPs or CIDR blocks
// (e.g. "127.0.0.1, 10.0.0.0/8"). Entries that fail to parse are dropped, so a
// typo narrows trust rather than widening it; an empty or entirely unparsable
// value yields nil, which trusts nobody (fail-closed).
func ParseTrustedProxies(raw string) []netip.Prefix {
	var proxies []netip.Prefix
	for _, entry := range strings.Split(raw, ",") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		if prefix, err := netip.ParsePrefix(entry); err == nil {
			proxies = append(proxies, prefix.Masked())
			continue
		}
		if addr, ok := parseIP(entry); ok {
			proxies = append(proxies, netip.PrefixFrom(addr, addr.BitLen()))
		}
	}
	return proxies
}

// HostOf strips the port from an http.Request.RemoteAddr-shaped address,
// tolerating bracketed/zoned IPv6 and the portless forms that tests and
// unix-socket peers produce.
func HostOf(remoteAddr string) string {
	if host, _, err := net.SplitHostPort(remoteAddr); err == nil {
		return host
	}
	return strings.TrimSpace(remoteAddr)
}

// IsTrusted reports whether addr falls inside one of the trusted prefixes.
func IsTrusted(addr netip.Addr, trusted []netip.Prefix) bool {
	for _, p := range trusted {
		if p.Contains(addr) {
			return true
		}
	}
	return false
}

// parseIP parses s as an IP literal and normalizes it: IPv4-mapped IPv6
// (::ffff:10.0.0.5) becomes plain IPv4 and any zone (fe80::1%eth0) is
// dropped, so trust checks and stored values see one canonical form.
func parseIP(s string) (netip.Addr, bool) {
	addr, err := netip.ParseAddr(strings.TrimSpace(s))
	if err != nil {
		return netip.Addr{}, false
	}
	return addr.Unmap().WithZone(""), true
}

// ClientIP resolves the client address for r:
//   - an untrusted immediate peer (RemoteAddr, port stripped) is the answer —
//     proxy headers are client-supplied and are never honored from it;
//   - a trusted peer's X-Forwarded-For (every header line, joined) is walked
//     right to left. Trusted hops are skipped; the first hop that is not a
//     trusted proxy is the trust boundary and is returned. A hop that does not
//     parse is also the boundary: it cannot be trusted, so the walk stops and
//     the peer is returned rather than skipping past it to a client-forged
//     entry further left;
//   - if X-Forwarded-For is absent, a trusted peer's X-Real-IP is used;
//   - otherwise the peer address is returned.
//
// The zero Addr is returned when RemoteAddr itself does not parse. Results are
// normalized (Unmap, no zone); raw header text never leaks to the caller.
func ClientIP(r *http.Request, trusted []netip.Prefix) netip.Addr {
	if r == nil {
		return netip.Addr{}
	}
	peer, peerOK := parseIP(HostOf(r.RemoteAddr))
	if !peerOK {
		return netip.Addr{}
	}
	if !IsTrusted(peer, trusted) {
		return peer
	}
	if xff := strings.Join(r.Header.Values("X-Forwarded-For"), ","); strings.TrimSpace(xff) != "" {
		hops := strings.Split(xff, ",")
		for i := len(hops) - 1; i >= 0; i-- {
			addr, ok := parseIP(hops[i])
			if !ok {
				return peer
			}
			if IsTrusted(addr, trusted) {
				continue
			}
			return addr
		}
		// Every hop was a trusted proxy: the chain started inside the trust
		// boundary, so the nearest hop we can attribute is the peer itself.
		return peer
	}
	if realIP := r.Header.Get("X-Real-IP"); strings.TrimSpace(realIP) != "" {
		if addr, ok := parseIP(realIP); ok {
			return addr
		}
	}
	return peer
}

// ClientIPString is ClientIP rendered as a bare IP literal, or "" when no
// address resolves — never a host:port pair and never unparsed header text,
// so it can be stored directly in an inet column.
func ClientIPString(r *http.Request, trusted []netip.Prefix) string {
	addr := ClientIP(r, trusted)
	if !addr.IsValid() {
		return ""
	}
	return addr.String()
}

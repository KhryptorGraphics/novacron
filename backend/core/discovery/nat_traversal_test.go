package discovery

import (
	"bytes"
	"encoding/binary"
	"net"
	"strconv"
	"strings"
	"testing"
	"time"

	"go.uber.org/zap"
)

var loopback = net.IPv4(127, 0, 0, 1)

// encodeAddressAttr builds a (XOR-)MAPPED-ADDRESS attribute as RFC 5389 §15.1-2
// specifies, independently of the parser under test.
func encodeAddressAttr(attrType uint16, ip net.IP, port int, txID [12]byte) STUNAttribute {
	family, raw := byte(0x01), ip.To4()
	if raw == nil {
		family, raw = 0x02, ip.To16()
	}
	xor := attrType == ATTR_XOR_MAPPED_ADDRESS

	value := make([]byte, 4+len(raw))
	value[1] = family
	p := uint16(port)
	if xor {
		p ^= uint16(STUN_MAGIC_COOKIE >> 16)
	}
	binary.BigEndian.PutUint16(value[2:4], p)

	var key [16]byte
	binary.BigEndian.PutUint32(key[:4], STUN_MAGIC_COOKIE)
	copy(key[4:], txID[:])
	for i, b := range raw {
		if xor {
			b ^= key[i]
		}
		value[4+i] = b
	}
	return STUNAttribute{Type: attrType, Length: uint16(len(value)), Value: value}
}

func stunDatagram(msgType uint16, txID [12]byte, attrs ...STUNAttribute) []byte {
	return (&STUNClient{}).marshalSTUNMessage(&STUNMessage{
		Type:          msgType,
		MagicCookie:   STUN_MAGIC_COOKIE,
		TransactionID: txID,
		Attributes:    attrs,
	})
}

func TestParseAddressAttribute(t *testing.T) {
	txID := [12]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	cases := []struct {
		name     string
		attr     STUNAttribute
		wantIP   net.IP
		wantPort int
		wantErr  string
	}{
		{name: "IPv4 MAPPED-ADDRESS", attr: encodeAddressAttr(ATTR_MAPPED_ADDRESS, net.IPv4(192, 168, 1, 1), 12345, txID), wantIP: net.IPv4(192, 168, 1, 1), wantPort: 12345},
		{name: "IPv4 XOR-MAPPED-ADDRESS", attr: encodeAddressAttr(ATTR_XOR_MAPPED_ADDRESS, net.IPv4(192, 168, 1, 1), 12345, txID), wantIP: net.IPv4(192, 168, 1, 1), wantPort: 12345},
		{name: "IPv6 MAPPED-ADDRESS", attr: encodeAddressAttr(ATTR_MAPPED_ADDRESS, net.ParseIP("2001:db8::1"), 8080, txID), wantIP: net.ParseIP("2001:db8::1"), wantPort: 8080},
		{name: "IPv6 XOR-MAPPED-ADDRESS uses transaction ID", attr: encodeAddressAttr(ATTR_XOR_MAPPED_ADDRESS, net.ParseIP("2001:db8::1"), 8080, txID), wantIP: net.ParseIP("2001:db8::1"), wantPort: 8080},
		{name: "attribute too short", attr: STUNAttribute{Type: ATTR_MAPPED_ADDRESS, Length: 4, Value: []byte{0, 1, 0, 80}}, wantErr: "address attribute too short"},
		{name: "unknown family", attr: STUNAttribute{Type: ATTR_MAPPED_ADDRESS, Length: 8, Value: []byte{0, 3, 0, 80, 1, 2, 3, 4}}, wantErr: "unsupported address family: 3"},
		{name: "IPv6 family with IPv4-sized value", attr: STUNAttribute{Type: ATTR_MAPPED_ADDRESS, Length: 8, Value: []byte{0, 2, 0, 80, 1, 2, 3, 4}}, wantErr: "IPv6 address attribute too short"},
	}

	sc := &STUNClient{}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			original := append([]byte(nil), tc.attr.Value...)
			endpoint, err := sc.parseAddressAttribute(&tc.attr, tc.attr.Type == ATTR_XOR_MAPPED_ADDRESS, txID)
			if tc.wantErr != "" {
				if err == nil || err.Error() != tc.wantErr {
					t.Fatalf("err = %v, want %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			if !endpoint.IP.Equal(tc.wantIP) || endpoint.Port != tc.wantPort {
				t.Errorf("endpoint = %s:%d, want %s:%d", endpoint.IP, endpoint.Port, tc.wantIP, tc.wantPort)
			}
			if !bytes.Equal(tc.attr.Value, original) {
				t.Errorf("parsing mutated the attribute: % x -> % x", original, tc.attr.Value)
			}
		})
	}
}

func TestSTUNMessageEncoding(t *testing.T) {
	sc := &STUNClient{}
	txID := [12]byte{9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2}

	t.Run("round trip pads attributes to 32-bit boundaries", func(t *testing.T) {
		odd := STUNAttribute{Type: 0x8022, Length: 5, Value: []byte("novac")}
		mapped := encodeAddressAttr(ATTR_XOR_MAPPED_ADDRESS, loopback, 3478, txID)
		data := stunDatagram(STUN_BINDING_RESPONSE, txID, odd, mapped)

		// 20-byte header + (4+5+3 padding) + (4+8).
		if len(data) != 20+12+12 || binary.BigEndian.Uint16(data[2:4]) != 24 {
			t.Fatalf("len=%d header length=%d, want 44/24", len(data), binary.BigEndian.Uint16(data[2:4]))
		}

		msg, err := sc.parseSTUNMessage(data)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if msg.Type != STUN_BINDING_RESPONSE || msg.TransactionID != txID || len(msg.Attributes) != 2 {
			t.Fatalf("message = %+v", msg)
		}
		for i, want := range []STUNAttribute{odd, mapped} {
			got := msg.Attributes[i]
			if got.Type != want.Type || got.Length != want.Length || !bytes.Equal(got.Value, want.Value) {
				t.Errorf("attribute %d = %+v, want %+v", i, got, want)
			}
		}

		endpoint, err := sc.extractExternalAddress(msg)
		if err != nil || !endpoint.IP.Equal(loopback) || endpoint.Port != 3478 {
			t.Errorf("extracted %v, %v; want 127.0.0.1:3478", endpoint, err)
		}
	})

	t.Run("truncated attribute is dropped", func(t *testing.T) {
		data := stunDatagram(STUN_BINDING_RESPONSE, txID, encodeAddressAttr(ATTR_MAPPED_ADDRESS, loopback, 1, txID))
		msg, err := sc.parseSTUNMessage(data[:len(data)-2])
		if err != nil || len(msg.Attributes) != 0 {
			t.Fatalf("msg=%+v err=%v, want no attributes", msg, err)
		}
		if _, err := sc.extractExternalAddress(msg); err == nil || err.Error() != "no mapped address found in STUN response" {
			t.Errorf("err = %v", err)
		}
	})

	for _, tc := range []struct {
		name    string
		data    []byte
		wantErr string
	}{
		{name: "shorter than header", data: make([]byte, 19), wantErr: "STUN message too short"},
		{name: "wrong magic cookie", data: make([]byte, 20), wantErr: "invalid STUN magic cookie"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := sc.parseSTUNMessage(tc.data); err == nil || err.Error() != tc.wantErr {
				t.Errorf("err = %v, want %q", err, tc.wantErr)
			}
		})
	}
}

// stunResponder builds the datagrams a fake server sends back for one request.
type stunResponder func(req *STUNMessage, from *net.UDPAddr) [][]byte

// startSTUNServer runs a loopback STUN server driven by respond.
func startSTUNServer(t *testing.T, respond stunResponder) STUNServer {
	t.Helper()
	conn, err := net.ListenUDP("udp", &net.UDPAddr{IP: loopback})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })

	go func() {
		parser := &STUNClient{}
		buf := make([]byte, 1500)
		for {
			n, from, err := conn.ReadFromUDP(buf)
			if err != nil {
				return
			}
			req, err := parser.parseSTUNMessage(buf[:n])
			if err != nil || req.Type != STUN_BINDING_REQUEST {
				continue
			}
			for _, datagram := range respond(req, from) {
				conn.WriteToUDP(datagram, from)
			}
		}
	}()
	return STUNServer{Host: "127.0.0.1", Port: conn.LocalAddr().(*net.UDPAddr).Port}
}

// reflector answers with the address it observed, like a STUN server with no NAT
// in between. portShift simulates a NAT that maps each destination differently.
func reflector(portShift int) stunResponder {
	return func(req *STUNMessage, from *net.UDPAddr) [][]byte {
		attr := encodeAddressAttr(ATTR_XOR_MAPPED_ADDRESS, from.IP, from.Port+portShift, req.TransactionID)
		return [][]byte{stunDatagram(STUN_BINDING_RESPONSE, req.TransactionID, attr)}
	}
}

// silentServer is a loopback UDP port that never answers.
func silentServer(t *testing.T) (STUNServer, *net.UDPConn) {
	t.Helper()
	conn, err := net.ListenUDP("udp", &net.UDPAddr{IP: loopback})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	return STUNServer{Host: "127.0.0.1", Port: conn.LocalAddr().(*net.UDPAddr).Port}, conn
}

func newLoopbackSTUNClient(servers ...STUNServer) *STUNClient {
	sc := NewSTUNClient(servers, zap.NewNop())
	sc.localAddr = &net.UDPAddr{IP: loopback}
	sc.timeout = 300 * time.Millisecond
	return sc
}

func serverAddr(s STUNServer) string {
	return net.JoinHostPort(s.Host, strconv.Itoa(s.Port))
}

func TestDiscoverExternalAddress(t *testing.T) {
	t.Run("returns the address the server observed", func(t *testing.T) {
		observed := make(chan *net.UDPAddr, 1)
		reflect := reflector(0)
		server := startSTUNServer(t, func(req *STUNMessage, from *net.UDPAddr) [][]byte {
			observed <- from
			return reflect(req, from)
		})

		endpoint, err := newLoopbackSTUNClient(server).DiscoverExternalAddress()
		if err != nil {
			t.Fatalf("DiscoverExternalAddress: %v", err)
		}
		from := <-observed
		if !endpoint.IP.Equal(loopback) || endpoint.Port != from.Port {
			t.Errorf("endpoint = %s:%d, want %s", endpoint.IP, endpoint.Port, from)
		}
		if endpoint.ServerUsed != serverAddr(server) || endpoint.LastUpdated.IsZero() {
			t.Errorf("ServerUsed=%q LastUpdated=%v", endpoint.ServerUsed, endpoint.LastUpdated)
		}
	})

	t.Run("ignores replies to other transactions", func(t *testing.T) {
		server := startSTUNServer(t, func(req *STUNMessage, from *net.UDPAddr) [][]byte {
			stale := req.TransactionID
			stale[0] ^= 0xff
			return [][]byte{
				stunDatagram(STUN_BINDING_RESPONSE, stale, encodeAddressAttr(ATTR_XOR_MAPPED_ADDRESS, net.IPv4(203, 0, 113, 9), 1, stale)),
				[]byte("not stun"),
				reflector(0)(req, from)[0],
			}
		})

		endpoint, err := newLoopbackSTUNClient(server).DiscoverExternalAddress()
		if err != nil {
			t.Fatalf("DiscoverExternalAddress: %v", err)
		}
		if !endpoint.IP.Equal(loopback) {
			t.Errorf("accepted a reply to another transaction: %s:%d", endpoint.IP, endpoint.Port)
		}
	})

	t.Run("falls back past silent and invalid servers", func(t *testing.T) {
		silent, _ := silentServer(t)
		invalid := STUNServer{Host: "127.0.0.1", Port: 70000}
		server := startSTUNServer(t, reflector(0))

		endpoint, err := newLoopbackSTUNClient(silent, invalid, server).DiscoverExternalAddress()
		if err != nil {
			t.Fatalf("DiscoverExternalAddress: %v", err)
		}
		if endpoint.ServerUsed != serverAddr(server) {
			t.Errorf("ServerUsed = %q, want %q", endpoint.ServerUsed, serverAddr(server))
		}
	})

	t.Run("error response fails the server", func(t *testing.T) {
		server := startSTUNServer(t, func(req *STUNMessage, _ *net.UDPAddr) [][]byte {
			return [][]byte{stunDatagram(STUN_ERROR_RESPONSE, req.TransactionID)}
		})

		_, err := newLoopbackSTUNClient(server).DiscoverExternalAddress()
		if err == nil || !strings.Contains(err.Error(), "all STUN servers failed") || !strings.Contains(err.Error(), "STUN server returned error") {
			t.Fatalf("err = %v", err)
		}
	})
}

func TestDetectNATType(t *testing.T) {
	cases := []struct {
		name    string
		servers func(*testing.T) []STUNServer
		want    int
		wantErr bool
	}{
		{
			name: "same mapping from both servers is a cone NAT",
			servers: func(t *testing.T) []STUNServer {
				return []STUNServer{startSTUNServer(t, reflector(0)), startSTUNServer(t, reflector(0))}
			},
			want: NAT_TYPE_FULL_CONE,
		},
		{
			name: "per-destination mapping is a symmetric NAT",
			servers: func(t *testing.T) []STUNServer {
				return []STUNServer{startSTUNServer(t, reflector(0)), startSTUNServer(t, reflector(1))}
			},
			want: NAT_TYPE_SYMMETRIC,
		},
		{
			name: "second server compared is the one after the first that answered",
			servers: func(t *testing.T) []STUNServer {
				silent, _ := silentServer(t)
				return []STUNServer{silent, startSTUNServer(t, reflector(0)), startSTUNServer(t, reflector(0))}
			},
			want: NAT_TYPE_FULL_CONE,
		},
		{
			name: "single server cannot classify",
			servers: func(t *testing.T) []STUNServer {
				return []STUNServer{startSTUNServer(t, reflector(0))}
			},
			want: NAT_TYPE_UNKNOWN,
		},
		{
			name: "no server answers",
			servers: func(t *testing.T) []STUNServer {
				silent, _ := silentServer(t)
				return []STUNServer{silent}
			},
			want:    NAT_TYPE_UNKNOWN,
			wantErr: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			detector := NewNATTypeDetector(newLoopbackSTUNClient(tc.servers(t)...), zap.NewNop())
			got, err := detector.DetectNATType()
			if (err != nil) != tc.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tc.wantErr)
			}
			if got != tc.want {
				t.Errorf("NAT type = %d, want %d", got, tc.want)
			}
		})
	}
}

func newLoopbackPuncher(t *testing.T) *UDPHolePuncher {
	t.Helper()
	uhp, err := NewUDPHolePuncher(&net.UDPAddr{IP: loopback}, zap.NewNop())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(uhp.Stop)
	return uhp
}

// readDatagram waits for one datagram on conn.
func readDatagram(t *testing.T, conn *net.UDPConn, timeout time.Duration) string {
	t.Helper()
	buf := make([]byte, 1500)
	conn.SetReadDeadline(time.Now().Add(timeout))
	n, _, err := conn.ReadFromUDP(buf)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	return string(buf[:n])
}

func TestHolePuncherReceiverProtocol(t *testing.T) {
	uhp := newLoopbackPuncher(t)
	cases := []struct {
		name string
		send string
		want string
	}{
		{name: "JSON handshake", send: `{"type":"HANDSHAKE","peer_id":"test-peer"}`, want: `{"type":"HANDSHAKE_ACK"}`},
		{name: "legacy handshake", send: "HANDSHAKE:legacy-peer", want: `{"type":"HANDSHAKE_ACK"}`},
		{name: "JSON ping echoes id", send: `{"type":"PING","id":42}`, want: `{"type":"PONG","id":42}`},
		{name: "legacy ping echoes id", send: "PING:1234", want: `{"type":"PONG","id":1234}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			conn, err := net.DialUDP("udp", nil, uhp.GetLocalAddress())
			if err != nil {
				t.Fatal(err)
			}
			defer conn.Close()

			if _, err := conn.Write([]byte(tc.send)); err != nil {
				t.Fatal(err)
			}
			if got := readDatagram(t, conn, 2*time.Second); got != tc.want {
				t.Errorf("reply = %s, want %s", got, tc.want)
			}
		})
	}
}

// snapshot reads a connection's mutable fields under the puncher's lock.
func snapshot(uhp *UDPHolePuncher, peerID string) (PeerConnection, bool) {
	uhp.mu.RLock()
	defer uhp.mu.RUnlock()
	conn, ok := uhp.connections[peerID]
	if !ok {
		return PeerConnection{}, false
	}
	return *conn, true
}

func TestEstablishConnectionOverLoopback(t *testing.T) {
	local, remote := newLoopbackPuncher(t), newLoopbackPuncher(t)
	remoteAddr := remote.GetLocalAddress()

	start := time.Now()
	conn, err := local.EstablishConnection("peer-b", remoteAddr)
	if err != nil {
		t.Fatalf("EstablishConnection: %v", err)
	}
	if elapsed := time.Since(start); elapsed >= local.handshakeInterval {
		t.Errorf("handshake took %v; the first ACK should complete it", elapsed)
	}

	got, ok := snapshot(local, "peer-b")
	if !ok || !got.Established || got.ConnectionType != "nat_traversal" || got.RemoteEndpoint.String() != remoteAddr.String() {
		t.Fatalf("registered connection = %+v (present=%v)", got, ok)
	}
	if again, err := local.EstablishConnection("peer-b", remoteAddr); err != nil || again != conn {
		t.Errorf("re-establishing returned %p, %v; want existing %p", again, err, conn)
	}

	quality, err := local.MeasureRTT(conn)
	if err != nil {
		t.Fatalf("MeasureRTT: %v", err)
	}
	if quality.PacketLoss != 0 || quality.RTT <= 0 || quality.RTT >= local.pingTimeout {
		t.Errorf("quality = %+v, want a PONG-based RTT without loss", quality)
	}

	if err := local.CloseConnection("peer-b"); err != nil {
		t.Fatalf("CloseConnection: %v", err)
	}
	if _, ok := local.GetConnection("peer-b"); ok {
		t.Error("connection still registered after close")
	}
	if err := local.CloseConnection("peer-b"); err == nil || err.Error() != "connection to peer peer-b not found" {
		t.Errorf("second close err = %v", err)
	}
}

func TestEstablishConnectionToSilentPeer(t *testing.T) {
	uhp := newLoopbackPuncher(t)
	uhp.handshakeInterval = 20 * time.Millisecond
	_, peer := silentServer(t)

	_, err := uhp.EstablishConnection("ghost", peer.LocalAddr().(*net.UDPAddr))
	if err == nil || err.Error() != "handshake failed: handshake timeout" {
		t.Fatalf("err = %v, want handshake timeout", err)
	}
	if _, ok := uhp.GetConnection("ghost"); ok {
		t.Error("failed handshake left a registered connection")
	}
	for i := range uhp.handshakeAttempts {
		if got := readDatagram(t, peer, time.Second); got != `{"type":"HANDSHAKE","peer_id":"ghost"}` {
			t.Fatalf("datagram %d = %s", i, got)
		}
	}
	peer.SetReadDeadline(time.Now().Add(50 * time.Millisecond))
	if _, _, err := peer.ReadFromUDP(make([]byte, 64)); err == nil {
		t.Errorf("more than %d handshake attempts were sent", uhp.handshakeAttempts)
	}
}

func TestMeasureRTTWithoutPongReportsLoss(t *testing.T) {
	uhp := newLoopbackPuncher(t)
	uhp.pingTimeout = 30 * time.Millisecond
	_, peer := silentServer(t)
	conn := &PeerConnection{PeerID: "quiet", RemoteEndpoint: peer.LocalAddr().(*net.UDPAddr), Quality: ConnectionQuality{RTT: 40 * time.Millisecond}}
	uhp.AddConnection(conn)

	quality, err := uhp.MeasureRTT(conn)
	if err != nil {
		t.Fatalf("MeasureRTT: %v", err)
	}
	if quality.PacketLoss <= 0 || quality.RTT <= 40*time.Millisecond {
		t.Errorf("quality = %+v, want packet loss and a degraded RTT", quality)
	}
}

func TestStopInterruptsInFlightOperations(t *testing.T) {
	cases := []struct {
		name  string
		start func(uhp *UDPHolePuncher, peer *net.UDPAddr) error
	}{
		{
			name: "handshake",
			start: func(uhp *UDPHolePuncher, peer *net.UDPAddr) error {
				_, err := uhp.EstablishConnection("ghost", peer)
				return err
			},
		},
		{
			name: "rtt measurement",
			start: func(uhp *UDPHolePuncher, peer *net.UDPAddr) error {
				conn := &PeerConnection{PeerID: "ghost", RemoteEndpoint: peer}
				uhp.AddConnection(conn)
				_, err := uhp.MeasureRTT(conn)
				return err
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			uhp := newLoopbackPuncher(t)
			_, peer := silentServer(t)

			done := make(chan error, 1)
			go func() { done <- tc.start(uhp, peer.LocalAddr().(*net.UDPAddr)) }()
			readDatagram(t, peer, 2*time.Second) // the operation is now waiting for a reply

			uhp.Stop()
			select {
			case err := <-done:
				if err == nil || !strings.Contains(err.Error(), "stopped") {
					t.Errorf("err = %v, want stopped", err)
				}
			case <-time.After(300 * time.Millisecond):
				t.Fatal("operation kept waiting after Stop")
			}
			if err := uhp.CloseConnection("ghost"); err == nil || err.Error() != "hole puncher is stopped" {
				t.Errorf("CloseConnection after Stop err = %v", err)
			}
		})
	}
}

func TestNATTraversalManagerOverLoopback(t *testing.T) {
	servers := []STUNServer{startSTUNServer(t, reflector(0)), startSTUNServer(t, reflector(0))}
	ntm, err := NewNATTraversalManager(&NATTraversalConfig{STUNServers: servers}, zap.NewNop())
	if err != nil {
		t.Fatal(err)
	}
	ntm.stunClient.timeout = 300 * time.Millisecond
	t.Cleanup(ntm.Stop)

	if err := ntm.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	external := ntm.GetExternalEndpoint()
	if !external.IP.Equal(loopback) || external.Port == 0 {
		t.Errorf("external endpoint = %s:%d", external.IP, external.Port)
	}
	if external.NATType != NAT_TYPE_FULL_CONE || ntm.GetNATType() != NAT_TYPE_FULL_CONE {
		t.Errorf("NAT type endpoint=%d manager=%d, want cone", external.NATType, ntm.GetNATType())
	}

	peer := newLoopbackPuncher(t)
	conn, err := ntm.EstablishP2PConnection("peer", peer.GetLocalAddress())
	if err != nil {
		t.Fatalf("EstablishP2PConnection: %v", err)
	}
	if conn.ConnectionType != "nat_traversal" {
		t.Errorf("ConnectionType = %q, want nat_traversal", conn.ConnectionType)
	}
	if active := ntm.GetActiveConnections(); active["peer"] != conn {
		t.Errorf("active connections = %v", active)
	}
}

func TestDiscoveryConnectsPeerThroughHolePuncher(t *testing.T) {
	ntm, err := NewNATTraversalManager(&NATTraversalConfig{STUNServers: []STUNServer{startSTUNServer(t, reflector(0))}}, zap.NewNop())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(ntm.Stop)
	service, err := NewInternetDiscovery(InternetDiscoveryConfig{Config: Config{NodeID: "local", NodeName: "local", NodeRole: "worker", Address: "127.0.0.1"}}, zap.NewNop())
	if err != nil {
		t.Fatal(err)
	}
	service.natTraversal = ntm

	peer := newLoopbackPuncher(t)
	addr := peer.GetLocalAddress()
	if err := service.ConnectToPeer(PeerInfo{NodeInfo: NodeInfo{ID: "peer"}, ExternalAddr: &ExternalEndpoint{IP: addr.IP, Port: addr.Port}}); err != nil {
		t.Fatalf("ConnectToPeer: %v", err)
	}
	service.connectionsMutex.RLock()
	connType := service.peerConnections["peer"].ConnectionType
	service.connectionsMutex.RUnlock()
	if connType != "nat_traversal" {
		t.Errorf("connection type = %q, want nat_traversal", connType)
	}

	// The puncher's shared socket must still carry this peer's traffic.
	conn, ok := ntm.holePuncher.GetConnection("peer")
	if !ok {
		t.Fatal("hole puncher lost the connection")
	}
	quality, err := ntm.holePuncher.MeasureRTT(conn)
	if err != nil || quality.PacketLoss != 0 {
		t.Errorf("RTT probe after connect = %+v, %v; want a PONG", quality, err)
	}
}

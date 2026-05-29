// Command lb is the load balancer: a TCP listener that hands accepted client
// fds to the API workers over Unix sockets with SCM_RIGHTS. No HTTP byte
// proxying — once SendFD returns, the API owns the connection end-to-end.
//
// Single-thread epoll. Usage:
//
//	lb <port> <uds_path1> [uds_path2 ...]
package main

import (
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"time"

	"github.com/vinicius-piassa/rinha-backend-2026-go/internal/netx"
	"golang.org/x/sys/unix"
)

const (
	maxBackends    = 8
	maxEvents      = 128
	backendRetries = 50
	retrySleep     = 100 * time.Millisecond

	epollIn     = 0x001
	epollEt     = 0x80000000
	tcpFastOpen = 23 // server-side TFO accept-queue length
)

var (
	backendsFD []int
	rrCursor   int
)

// listenTCP binds 0.0.0.0:port, applies the latency sockopts, listens(4096),
// and sets O_NONBLOCK.
func listenTCP(port int) (int, error) {
	fd, err := unix.Socket(unix.AF_INET, unix.SOCK_STREAM|unix.SOCK_CLOEXEC, 0)
	if err != nil {
		return -1, err
	}
	// best-effort latency/throughput knobs (failures are harmless)
	unix.SetsockoptInt(fd, unix.SOL_SOCKET, unix.SO_REUSEADDR, 1)
	unix.SetsockoptInt(fd, unix.SOL_SOCKET, unix.SO_REUSEPORT, 1)
	unix.SetsockoptInt(fd, unix.SOL_TCP, unix.TCP_DEFER_ACCEPT, 1)
	unix.SetsockoptInt(fd, unix.SOL_SOCKET, unix.SO_BUSY_POLL, 50)
	unix.SetsockoptInt(fd, unix.SOL_SOCKET, unix.SO_PREFER_BUSY_POLL, 1)
	unix.SetsockoptInt(fd, unix.SOL_SOCKET, unix.SO_BUSY_POLL_BUDGET, 8)
	unix.SetsockoptInt(fd, unix.SOL_TCP, tcpFastOpen, 256)

	addr := &unix.SockaddrInet4{Port: port} // Addr zero = INADDR_ANY
	if err := unix.Bind(fd, addr); err != nil {
		unix.Close(fd)
		return -1, err
	}
	if err := unix.Listen(fd, 4096); err != nil {
		unix.Close(fd)
		return -1, err
	}
	unix.SetNonblock(fd, true)
	return fd, nil
}

// acceptLoop drains accept4 until EAGAIN; each client fd is round-robined to a
// backend via SCM_RIGHTS, then closed locally.
func acceptLoop(listenFD int) {
	for {
		cfd, _, err := unix.Accept4(listenFD, unix.SOCK_CLOEXEC)
		if err == unix.EINTR {
			continue
		}
		if err == unix.EAGAIN || err == unix.EWOULDBLOCK {
			return
		}
		if err != nil {
			return
		}
		be := backendsFD[rrCursor%len(backendsFD)]
		rrCursor++
		_ = netx.SendFD(be, cfd) // ignore error — proceed to close
		unix.Close(cfd)
	}
}

// selfWarm opens a few short connections back to the LB's own listen port so
// the docker-proxy / NAT / accept→SCM_RIGHTS→API path is primed before real
// traffic arrives. Runs in a transient goroutine (fork after the runtime
// starts is unsafe), warming the same kernel paths.
func selfWarm(port int) {
	const iters = 32
	body := []byte("POST /fraud-score HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 407\r\n\r\n" +
		`{"id":"tx-warm","transaction":{"amount":384.88,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},"customer":{"avg_amount":769.76,"tx_count_24h":3,"known_merchants":["MERC-009","MERC-001"]},"merchant":{"id":"MERC-001","mcc":"5912","avg_amount":298.95},"terminal":{"is_online":false,"card_present":true,"km_from_home":13.7},"last_transaction":{"timestamp":"2026-03-11T14:58:35Z","km_from_current":18.8}}`)
	addr := &unix.SockaddrInet4{Port: port, Addr: [4]byte{127, 0, 0, 1}}
	var scratch [4096]byte
	for i := 0; i < iters; i++ {
		fd, err := unix.Socket(unix.AF_INET, unix.SOCK_STREAM|unix.SOCK_CLOEXEC, 0)
		if err != nil {
			continue
		}
		if unix.Connect(fd, addr) == nil {
			unix.Sendmsg(fd, body, nil, nil, unix.MSG_NOSIGNAL)
			unix.Read(fd, scratch[:])
		}
		unix.Close(fd)
	}
}

func serverLoop(epfd, listenFD int) {
	events := make([]unix.EpollEvent, maxEvents)
	for {
		n, err := unix.EpollWait(epfd, events, -1) // blocking; no spin at 0.05 CPU
		if err == unix.EINTR {
			continue
		}
		if err != nil {
			continue
		}
		for i := 0; i < n; i++ {
			if int(events[i].Fd) == listenFD {
				acceptLoop(listenFD)
			}
		}
	}
}

func die(msg string) {
	os.Stderr.WriteString(msg + "\n")
	os.Exit(1)
}

func main() {
	runtime.GOMAXPROCS(1)
	// GC off: the accept→SCM_RIGHTS→close path allocates nothing per request.
	// SetMemoryLimit is a backstop well under the LB's tiny budget.
	debug.SetGCPercent(-1)
	debug.SetMemoryLimit(6 << 20)

	if len(os.Args) < 3 {
		die("usage: lb <port> <uds_path1> [uds_path2 ...]")
	}
	port, _ := strconv.Atoi(os.Args[1])

	unix.Prctl(unix.PR_SET_TIMERSLACK, 1, 0, 0, 0)
	// No mlockall here: at the LB's tiny memory budget, pinning the whole Go
	// runtime resident is counterproductive. We rely on the pages staying hot
	// under steady traffic instead.

	listenFD, err := listenTCP(port)
	if err != nil {
		die("lb: listen_tcp failed: " + err.Error())
	}

	paths := os.Args[2:]
	if len(paths) > maxBackends {
		paths = paths[:maxBackends]
	}
	for _, p := range paths {
		bfd := -1
		for r := 0; r < backendRetries; r++ {
			fd, e := unix.Socket(unix.AF_UNIX, unix.SOCK_STREAM|unix.SOCK_CLOEXEC, 0)
			if e == nil {
				if unix.Connect(fd, &unix.SockaddrUnix{Name: p}) == nil {
					bfd = fd
					break
				}
				unix.Close(fd)
			}
			time.Sleep(retrySleep)
		}
		if bfd < 0 {
			die("lb: backend connect failed (gave up): " + p)
		}
		backendsFD = append(backendsFD, bfd)
	}

	epfd, err := unix.EpollCreate1(unix.EPOLL_CLOEXEC)
	if err != nil {
		die("lb: epoll_create1 failed")
	}
	netx.SetEpollBusyPoll(epfd)
	if err := unix.EpollCtl(epfd, unix.EPOLL_CTL_ADD, listenFD,
		&unix.EpollEvent{Events: epollIn | epollEt, Fd: int32(listenFD)}); err != nil {
		die("lb: epoll_ctl add listen failed")
	}

	go selfWarm(port)
	serverLoop(epfd, listenFD)
}

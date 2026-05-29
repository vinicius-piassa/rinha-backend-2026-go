// Command server is the API worker. It receives client TCP fds from the LB
// over a Unix control socket (SCM_RIGHTS), then runs an epoll event loop that
// frames HTTP requests and runs the fraud pipeline:
//
//	recv → frame HTTP → parse JSON → vectorize → route by tag → IVF k-NN → reply
//
// Usage: server <uds_path>
package main

import (
	"bytes"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"unsafe"

	"github.com/vinicius-piassa/rinha-backend-2026-go/internal/fraud"
	"github.com/vinicius-piassa/rinha-backend-2026-go/internal/index"
	"github.com/vinicius-piassa/rinha-backend-2026-go/internal/netx"
	"golang.org/x/sys/unix"
)

const (
	bufSize   = 4096
	maxFDs    = 1024
	maxEvents = 128

	epollIn     = 0x001
	epollRdhup  = 0x2000
	schedFIFO   = 1
	workerRTPri = 10
)

// connState is the per-fd receive buffer, indexed by fd number.
type connState struct {
	buf [bufSize]byte
	pos int
}

var (
	states  []connState
	ctrlFD  int
	epollFD int

	// partition indices, routed by the 4-bit tag
	// (card_present<<3 | is_online<<2 | unknown_merchant<<1 | has_last_tx).
	indices [index.NPartitions]*index.IvfIndex

	// preallocated framing needles (avoid per-call []byte(string) conversions)
	hdrSep = []byte("\r\n\r\n")
	clKey  = []byte("content-length:")

	// pre-rendered responses, one per fraud-score bucket (count 0..5).
	responses = [6][]byte{
		[]byte("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}"),
		[]byte("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}"),
		[]byte("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}"),
		[]byte("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}"),
		[]byte("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}"),
		[]byte("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}"),
	}
	readyResp = []byte("HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
	errResp   = []byte("HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
)

// bindControlUDS unlinks any stale socket, binds path, listens, and blocks in
// accept4 until the LB connects, returning the accepted control fd.
func bindControlUDS(path string) (int, error) {
	unix.Unlink(path) // best-effort
	fd, err := unix.Socket(unix.AF_UNIX, unix.SOCK_STREAM|unix.SOCK_CLOEXEC, 0)
	if err != nil {
		return -1, err
	}
	if err := unix.Bind(fd, &unix.SockaddrUnix{Name: path}); err != nil {
		unix.Close(fd)
		return -1, err
	}
	unix.Chmod(path, 0o666) // LB usually runs as a different uid; best-effort
	if err := unix.Listen(fd, 8); err != nil {
		unix.Close(fd)
		return -1, err
	}
	for {
		cfd, _, err := unix.Accept4(fd, unix.SOCK_CLOEXEC)
		if err == unix.EINTR {
			continue
		}
		unix.Close(fd)
		if err != nil {
			return -1, err
		}
		return cfd, nil
	}
}

// handleRequest routes one framed HTTP request. POST /fraud-score runs the
// full pipeline; GET returns the ready response; anything else is a 400.
// req is the full framed message; bodyOff is the start of the JSON body.
func handleRequest(req []byte, bodyOff int) []byte {
	n := len(req)
	if n >= 5 && req[0] == 'P' && req[1] == 'O' && req[2] == 'S' && req[3] == 'T' && req[4] == ' ' {
		var r fraud.Request
		if !fraud.ParseRequest(req[bodyOff:], &r) {
			return errResp
		}
		v := fraud.Vectorize(&r)
		// 4-bit tag: has_last | unknown | online | card. The online&card combo
		// never occurs in training, so its partitions are absent — fall back by
		// clearing the card bit (online&!card is always populated).
		tag := 0
		if r.HasLastTx {
			tag |= 1
		}
		if !r.KnownMerchant {
			tag |= 2
		}
		if r.IsOnline {
			tag |= 4
		}
		if r.CardPresent {
			tag |= 8
		}
		if indices[tag] == nil {
			tag &^= 8
			if indices[tag] == nil {
				tag &^= 4
			}
		}
		cnt := indices[tag].Search(&v)
		if cnt > 5 {
			cnt = 5
		}
		return responses[cnt]
	}
	if n >= 4 && req[0] == 'G' && req[1] == 'E' && req[2] == 'T' && req[3] == ' ' {
		return readyResp
	}
	return errResp
}

func sendAll(fd int, p []byte) error {
	off := 0
	for off < len(p) {
		n, err := unix.SendmsgN(fd, p[off:], nil, nil, unix.MSG_NOSIGNAL)
		if err == unix.EINTR {
			continue
		}
		if err != nil {
			return err
		}
		off += n
	}
	return nil
}

// schedParam mirrors `struct sched_param` (a single priority int).
type schedParam struct{ priority int32 }

// setRealtimePriority promotes this thread to SCHED_FIFO so an inbound packet
// wakes us above SCHED_OTHER. Best-effort (needs CAP_SYS_NICE).
func setRealtimePriority() {
	p := schedParam{priority: workerRTPri}
	unix.Syscall(unix.SYS_SCHED_SETSCHEDULER, 0, uintptr(schedFIFO), uintptr(unsafe.Pointer(&p)))
}

func closeClient(fd int) {
	unix.EpollCtl(epollFD, unix.EPOLL_CTL_DEL, fd, nil)
	unix.Close(fd)
	if fd < maxFDs {
		states[fd].pos = 0
	}
}

// contentLength scans header bytes for "content-length:" (case-insensitive)
// and returns its value, or -1 if absent. Allocation-free.
func contentLength(hdr []byte) int {
	i := indexFold(hdr, clKey)
	if i < 0 {
		return -1
	}
	j := i + len(clKey)
	for j < len(hdr) && (hdr[j] == ' ' || hdr[j] == '\t') {
		j++
	}
	n := 0
	for j < len(hdr) && hdr[j] >= '0' && hdr[j] <= '9' {
		n = n*10 + int(hdr[j]-'0')
		j++
	}
	return n
}

// indexFold finds needle in hay, case-insensitive on ASCII, without allocating.
// needle must already be lowercase.
func indexFold(hay, needle []byte) int {
	if len(needle) == 0 {
		return 0
	}
	last := len(hay) - len(needle)
	for i := 0; i <= last; i++ {
		k := 0
		for ; k < len(needle); k++ {
			c := hay[i+k]
			if c >= 'A' && c <= 'Z' {
				c += 'a' - 'A'
			}
			if c != needle[k] {
				break
			}
		}
		if k == len(needle) {
			return i
		}
	}
	return -1
}

// recvNB does a non-blocking recvfrom with a NULL source address, avoiding the
// per-call sockaddr allocation that unix.Recvfrom incurs.
func recvNB(fd int, p []byte) (int, unix.Errno) {
	r0, _, e := unix.Syscall6(unix.SYS_RECVFROM, uintptr(fd),
		uintptr(unsafe.Pointer(&p[0])), uintptr(len(p)), uintptr(unix.MSG_DONTWAIT), 0, 0)
	return int(r0), e
}

func handleClientEvent(fd int) {
	st := &states[fd]
	if st.pos >= bufSize {
		closeClient(fd) // oversized request
		return
	}
	n, errno := recvNB(fd, st.buf[st.pos:])
	if errno == unix.EAGAIN || errno == unix.EWOULDBLOCK || errno == unix.EINTR {
		return
	}
	if n == 0 || errno != 0 {
		closeClient(fd)
		return
	}
	st.pos += n

	for st.pos > 0 {
		hdrEnd := bytes.Index(st.buf[:st.pos], hdrSep)
		if hdrEnd < 0 {
			return // partial headers — wait for more
		}
		bodyOff := hdrEnd + 4
		cl := contentLength(st.buf[:bodyOff])
		if cl < 0 {
			cl = 0
		}
		total := bodyOff + cl
		if st.pos < total {
			return // body incomplete — wait for more
		}
		if err := sendAll(fd, handleRequest(st.buf[:total], bodyOff)); err != nil {
			closeClient(fd)
			return
		}
		// shift any pipelined leftover to the front
		rem := st.pos - total
		if rem > 0 {
			copy(st.buf[:rem], st.buf[total:st.pos])
		}
		st.pos = rem
	}
}

var (
	ctrlOOB   [256]byte
	fdScratch = make([]int, 0, 64)
)

func handleCtrlEvent() {
	fds, ok, err := netx.RecvFDs(ctrlFD, ctrlOOB[:], fdScratch[:0])
	if !ok || err != nil {
		return
	}
	for _, fd := range fds {
		if fd >= maxFDs {
			unix.Close(fd) // out of state range — reject
			continue
		}
		states[fd].pos = 0
		unix.EpollCtl(epollFD, unix.EPOLL_CTL_ADD, fd,
			&unix.EpollEvent{Events: epollIn | epollRdhup, Fd: int32(fd)})
	}
}

func serverLoop() {
	events := make([]unix.EpollEvent, maxEvents)
	for {
		n, err := unix.EpollWait(epollFD, events, 1) // 1ms; kernel busy-polls 50µs first
		if err == unix.EINTR {
			continue
		}
		if n <= 0 {
			continue
		}
		for i := 0; i < n; i++ {
			fd := int(events[i].Fd)
			if fd == ctrlFD {
				handleCtrlEvent()
			} else {
				handleClientEvent(fd)
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
	// GC off: the steady-state per-request path is allocation-free, so periodic
	// GC would only burn CPU at our 0.475-core budget. SetMemoryLimit is a
	// backstop that triggers a collection only if memory creeps toward the cap.
	debug.SetGCPercent(-1)
	debug.SetMemoryLimit(160 << 20) // < 171 MB container limit

	if len(os.Args) < 2 {
		die("usage: server <uds_path> [index_dir]")
	}
	udsPath := os.Args[1]
	indexDir := "."
	if len(os.Args) >= 3 {
		indexDir = os.Args[2]
	}

	if !index.HasAVX2() {
		die("fatal: CPU sem AVX2")
	}

	unix.Prctl(unix.PR_SET_TIMERSLACK, 1, 0, 0, 0)
	unix.Mlockall(unix.MCL_CURRENT | unix.MCL_FUTURE)
	setRealtimePriority()

	states = make([]connState, maxFDs)

	// Open the partition indices (up to 16; the online&card tags don't exist,
	// so missing files are skipped — the router falls back for them).
	loaded := 0
	for i := 0; i < index.NPartitions; i++ {
		path := indexDir + "/index_p" + strconv.Itoa(i) + ".bin"
		if _, err := os.Stat(path); err != nil {
			continue // tag has no partition (e.g. online&card)
		}
		ix, err := index.Open(path)
		if err != nil {
			die("error: failed to open index_p" + strconv.Itoa(i) + ".bin: " + err.Error())
		}
		indices[i] = ix
		loaded++
	}
	if loaded == 0 {
		die("error: no index files found in " + indexDir)
	}

	cfd, err := bindControlUDS(udsPath)
	if err != nil {
		die("error: bind_control_uds failed: " + err.Error())
	}
	ctrlFD = cfd
	unix.SetNonblock(ctrlFD, true)

	epollFD, err = unix.EpollCreate1(unix.EPOLL_CLOEXEC)
	if err != nil {
		die("error: epoll_create1 failed")
	}
	netx.SetEpollBusyPoll(epollFD)
	if err := unix.EpollCtl(epollFD, unix.EPOLL_CTL_ADD, ctrlFD,
		&unix.EpollEvent{Events: epollIn, Fd: int32(ctrlFD)}); err != nil {
		die("error: epoll_ctl add ctrl failed")
	}

	serverLoop()
}

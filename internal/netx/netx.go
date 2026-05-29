// Package netx holds the low-level socket primitives shared by the LB and the
// API server: SCM_RIGHTS fd passing and the EPIOCSPARAMS epoll busy-poll knob.
// All pure Go over raw syscalls via golang.org/x/sys/unix — no cgo.
//
// The hot paths here are allocation-free: SendFD builds its control message in
// a stack array and RecvFDs walks the cmsg buffer in place into a caller-owned
// slice, so the GC never sees per-call garbage.
package netx

import (
	"unsafe"

	"golang.org/x/sys/unix"
)

// EPIOCSPARAMS sets epoll busy-poll parameters (Linux 6.9+).
const EPIOCSPARAMS = 0x40087001

// cmsg layout on 64-bit Linux: cmsg_len(8) | cmsg_level(4) | cmsg_type(4) | data.
// CMSG_LEN(4) = 20, CMSG_SPACE(4) = 24.
const (
	cmsgHdrLen   = 16
	cmsgLenOneFD = 20
	cmsgSpaceFD  = 24
)

// fByte is the 1-byte iov payload accompanying the passed fd. Package-level so
// SendFD allocates nothing per call.
var fByte = [1]byte{'F'}

// epollParams mirrors `struct epoll_params` (8 bytes, packed).
type epollParams struct {
	busyPollUsecs  uint32
	busyPollBudget uint16
	preferBusyPoll uint8
	_              uint8
}

// SetEpollBusyPoll asks the kernel to NAPI-busy-poll for 50µs inside every
// epoll_wait before sleeping. Best-effort: pre-6.9 kernels return ENOTTY and
// we silently fall back to standard wake semantics.
func SetEpollBusyPoll(epfd int) {
	p := epollParams{busyPollUsecs: 50, busyPollBudget: 8, preferBusyPoll: 1}
	unix.Syscall(unix.SYS_IOCTL, uintptr(epfd), uintptr(EPIOCSPARAMS), uintptr(unsafe.Pointer(&p)))
}

// SendFD passes clientFd to the peer of udsFd over a Unix socket via a
// SCM_RIGHTS cmsg, with a 1-byte 'F' payload. The receiver takes ownership;
// the caller should close clientFd afterward. Zero heap allocations.
func SendFD(udsFd, clientFd int) error {
	var oob [cmsgSpaceFD]byte
	*(*uint64)(unsafe.Pointer(&oob[0])) = cmsgLenOneFD
	*(*int32)(unsafe.Pointer(&oob[8])) = unix.SOL_SOCKET
	*(*int32)(unsafe.Pointer(&oob[12])) = unix.SCM_RIGHTS
	*(*int32)(unsafe.Pointer(&oob[16])) = int32(clientFd)
	for {
		_, err := unix.SendmsgN(udsFd, fByte[:], oob[:], nil, unix.MSG_NOSIGNAL)
		if err == unix.EINTR {
			continue
		}
		return err
	}
}

// RecvFDs does a single non-blocking recvmsg on ctrlFd and appends any
// SCM_RIGHTS-delivered fds to out (which the caller should pass with spare
// capacity so no allocation occurs). Returns ok=false on EOF; err carries a
// real error or EAGAIN so the edge-triggered caller can stop draining.
//
// oob must be large enough for the expected cmsg batch (e.g. 256 bytes).
func RecvFDs(ctrlFd int, oob []byte, out []int) (fds []int, ok bool, err error) {
	var p [64]byte
	var n, oobn int
	for {
		n, oobn, _, _, err = unix.Recvmsg(ctrlFd, p[:], oob, unix.MSG_CMSG_CLOEXEC|unix.MSG_DONTWAIT)
		if err == unix.EINTR {
			continue
		}
		break
	}
	if err != nil {
		return out, true, err // includes EAGAIN
	}
	if n == 0 {
		return out, false, nil // EOF
	}

	// Walk the cmsg buffer in place.
	buf := oob[:oobn]
	for off := 0; off+cmsgHdrLen <= len(buf); {
		clen := int(*(*uint64)(unsafe.Pointer(&buf[off])))
		if clen < cmsgHdrLen || off+clen > len(buf) {
			break
		}
		level := *(*int32)(unsafe.Pointer(&buf[off+8]))
		typ := *(*int32)(unsafe.Pointer(&buf[off+12]))
		if level == unix.SOL_SOCKET && typ == unix.SCM_RIGHTS {
			for d := off + cmsgHdrLen; d+4 <= off+clen; d += 4 {
				out = append(out, int(*(*int32)(unsafe.Pointer(&buf[d]))))
			}
		}
		off += (clen + 7) &^ 7 // advance, 8-byte aligned
	}
	return out, true, nil
}

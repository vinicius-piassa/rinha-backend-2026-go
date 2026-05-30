package index

import (
	"encoding/binary"
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/unix"
)

// Search/probe constants.
const (
	IdxBits         = 22
	CidBits         = 12
	CidMask         = 0xFFF
	NProbeInitial   = 12
	NProbeRepairMin = 1
	NProbeRepairMax = 4
)

// IvfIndex is one partition's index mapped into memory. The section slices
// alias the mmap'd file (zero-copy); bpsoa* are derived buffers.
type IvfIndex struct {
	data      []byte
	NClusters int
	NVectors  int

	clusterOffsets   []uint32 // len K+1
	bboxMin, bboxMax []int16  // K*16

	// pairs[p] is a []int16 view of the p-th packed pair section: element
	// 2*v holds dim 2p of vector v, element 2*v+1 holds dim 2p+1. Sized with
	// 16 elements of slack so a 16-lane SIMD load on the last vector stays in
	// the mapping (the file's section padding + 64-byte tail cover it).
	pairs [NPairs][]int16

	labels []uint8 // len NVectors

	bpsoaMin, bpsoaMax []int16 // nGroups * 7 * 16, 8-cluster SoA pair groups
}

func align64(x int) int { return (x + 63) &^ 63 }

// viewAt aliases n elements of type T starting at byte offset off in data.
func viewAt[T any](data []byte, off, n int) []T {
	return unsafe.Slice((*T)(unsafe.Pointer(&data[off])), n)
}

// Open mmaps the index file, validates the header, wires up the section
// slices, and builds the 8-cluster pair SoA used by phase 1.
func Open(path string) (*IvfIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := int(st.Size())
	data, err := unix.Mmap(int(f.Fd()), 0, size, unix.PROT_READ, unix.MAP_PRIVATE|unix.MAP_POPULATE)
	if err != nil {
		return nil, err
	}
	// best-effort: keep resident + back with hugepages
	unix.Mlock(data)
	unix.Madvise(data, unix.MADV_HUGEPAGE)
	unix.Madvise(data, unix.MADV_WILLNEED)

	if size < 64 || string(data[0:8]) != magic {
		return nil, fmt.Errorf("index: bad magic in %s", path)
	}
	nc := int(binary.LittleEndian.Uint32(data[8:12]))
	if nc < 1 || nc > NClusters {
		return nil, fmt.Errorf("index: n_clusters=%d, want 1..%d", nc, NClusters)
	}
	nv := int(binary.LittleEndian.Uint32(data[12:16]))

	ix := &IvfIndex{data: data, NClusters: nc, NVectors: nv}

	off := 64
	ix.clusterOffsets = viewAt[uint32](data, off, nc+1)
	off = align64(off + (nc+1)*4)
	ix.bboxMin = viewAt[int16](data, off, nc*16)
	off = align64(off + nc*32)
	ix.bboxMax = viewAt[int16](data, off, nc*16)
	off = align64(off + nc*32)
	for p := 0; p < NPairs; p++ {
		ix.pairs[p] = viewAt[int16](data, off, 2*nv+16)
		off = align64(off + nv*4)
	}
	ix.labels = viewAt[uint8](data, off, nv)
	off += nv
	if off > size {
		return nil, fmt.Errorf("index: sections overrun file (%d > %d)", off, size)
	}

	ix.buildBPSOA()
	return ix, nil
}

// Close unmaps the file.
func (ix *IvfIndex) Close() error {
	if ix.data != nil {
		err := unix.Munmap(ix.data)
		ix.data = nil
		return err
	}
	return nil
}

// buildBPSOA reshapes the flat per-cluster bbox arrays into 8-cluster pair
// groups so phase 1 can process 8 clusters per SIMD iteration. Phantom lanes
// (cluster index >= K) get INT16_MAX/MIN so their lower bound is never minimal.
func (ix *IvfIndex) buildBPSOA() {
	K := ix.NClusters
	nGroups := (K + 7) / 8
	ix.bpsoaMin = make([]int16, nGroups*7*16)
	ix.bpsoaMax = make([]int16, nGroups*7*16)
	for g := 0; g < nGroups; g++ {
		for p := 0; p < 7; p++ {
			dst := (g*7 + p) * 16
			for l := 0; l < 8; l++ {
				c := g*8 + l
				di := dst + l*2
				if c < K {
					ix.bpsoaMin[di] = ix.bboxMin[c*16+2*p]
					ix.bpsoaMin[di+1] = ix.bboxMin[c*16+2*p+1]
					ix.bpsoaMax[di] = ix.bboxMax[c*16+2*p]
					ix.bpsoaMax[di+1] = ix.bboxMax[c*16+2*p+1]
				} else {
					ix.bpsoaMin[di], ix.bpsoaMin[di+1] = 0x7FFF, 0x7FFF
					ix.bpsoaMax[di], ix.bpsoaMax[di+1] = -0x8000, -0x8000
				}
			}
		}
	}
}

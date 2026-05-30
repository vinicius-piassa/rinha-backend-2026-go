// Package index builds and (later) reads the IVF k-NN index files.
//
// On-disk layout (every section zero-padded up to 64 bytes, except the labels
// section which is written raw and followed by a 64-byte tail pad so a runtime
// SIMD over-read on the last batch can't cross the mapping boundary):
//
//	header           64 bytes      magic | n_clusters | n_vectors
//	cluster_offsets  (K+1)*4
//	bbox_min         K*16*2        int16[16] per cluster
//	bbox_max         K*16*2
//	pair_arr[0..6]   n*4 each      two quantized int16 dims packed per int32
//	labels           n             one byte per vector (1 = fraud)
//	tail pad         64
package index

import (
	"bufio"
	"encoding/binary"
	"os"
	"unsafe"
)

const (
	NDims      = 14
	NClusters  = 2048
	NPairs     = 7
	KNeighbors = 5
	Scale      = 10000

	// NPartitions is the 4-bit tag space (has_last | unknown | online | card).
	// Tags with both online and card set are empty in the corpus, so only 12
	// of the 16 files exist.
	NPartitions = 16

	magic = "RNH4-IDX"
)

// bytesOf reinterprets a slice of fixed-size values as a little-endian byte
// view (zero-copy). Safe on amd64; the on-disk format is little-endian.
func bytesOf[T any](s []T) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*int(unsafe.Sizeof(s[0])))
}

// writePadded writes b, then zeros to round the section length up to 64.
func writePadded(w *bufio.Writer, b []byte) error {
	if _, err := w.Write(b); err != nil {
		return err
	}
	if pad := (64 - len(b)%64) % 64; pad > 0 {
		var z [64]byte
		if _, err := w.Write(z[:pad]); err != nil {
			return err
		}
	}
	return nil
}

// WriteIndexBin serializes one partition's index to path.
func WriteIndexBin(
	path string,
	n int,
	clusterOffsets []uint32,
	bboxMin, bboxMax []int16,
	pairArr [NPairs][]int32,
	labels []uint8,
) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)

	var hdr [64]byte
	copy(hdr[0:8], magic)
	binary.LittleEndian.PutUint32(hdr[8:12], uint32(len(clusterOffsets)-1)) // actual K
	binary.LittleEndian.PutUint32(hdr[12:16], uint32(n))
	if err := writePadded(w, hdr[:]); err != nil {
		return err
	}
	if err := writePadded(w, bytesOf(clusterOffsets)); err != nil {
		return err
	}
	if err := writePadded(w, bytesOf(bboxMin)); err != nil {
		return err
	}
	if err := writePadded(w, bytesOf(bboxMax)); err != nil {
		return err
	}
	for p := 0; p < NPairs; p++ {
		if err := writePadded(w, bytesOf(pairArr[p])); err != nil {
			return err
		}
	}
	if _, err := w.Write(labels); err != nil { // raw, no section pad
		return err
	}
	var tail [64]byte
	if _, err := w.Write(tail[:]); err != nil {
		return err
	}
	if err := w.Flush(); err != nil {
		return err
	}
	return f.Close()
}

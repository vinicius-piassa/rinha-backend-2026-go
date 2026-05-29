package index

import (
	"math"
	"runtime"
	"sync"

	"simd/archsimd"
)

// Ref is one corpus vector. Lanes 14,15 of V are kept at 0 so the SIMD L2 over
// 16 lanes ignores them without a mask.
type Ref struct {
	V     [16]float32
	Label uint8
}

// Tag computes the 4-bit partition tag. All four bits are extreme-valued
// (0 or SCALE) dimensions, so two vectors differing in any of them are SCALE
// apart on that axis — neighbours never straddle the boundary, keeping the
// routing detection-safe (validated at 0/5000 verdict changes vs full search).
//
//	bit0 = has_last_tx      (V[5] >= 0)
//	bit1 = unknown_merchant (V[11] > 0.5)
//	bit2 = is_online        (V[9]  > 0.5)
//	bit3 = card_present     (V[10] > 0.5)
//
// is_online & card_present never co-occur in the corpus, so 4 of the 16 tags
// are empty; the builder skips them and the server falls back at routing.
func (r *Ref) Tag() int {
	tag := 0
	if r.V[5] >= 0 {
		tag |= 1
	}
	if r.V[11] > 0.5 {
		tag |= 2
	}
	if r.V[9] > 0.5 {
		tag |= 4
	}
	if r.V[10] > 0.5 {
		tag |= 8
	}
	return tag
}

// FilterByTag compacts refs in place, keeping only those whose tag matches.
func FilterByTag(refs []Ref, tag int) []Ref {
	dst := 0
	for src := range refs {
		if refs[src].Tag() == tag {
			if src != dst {
				refs[dst] = refs[src]
			}
			dst++
		}
	}
	return refs[:dst]
}

// quantize maps the 14 float dims to int16 (round-to-even × Scale, saturated).
// Lanes 14,15 stay 0.
func quantize(v *[16]float32) [16]int16 {
	var q [16]int16
	for d := 0; d < NDims; d++ {
		x := math.RoundToEven(float64(v[d]) * Scale)
		if x > math.MaxInt16 {
			x = math.MaxInt16
		} else if x < math.MinInt16 {
			x = math.MinInt16
		}
		q[d] = int16(x)
	}
	return q
}

// l2sqF32 is the squared L2 distance over 16 lanes (14 real + 2 zero pad).
// Compiles to VSUBPS + VMULPS + VADDPS on 256-bit registers.
func l2sqF32(a, b *[16]float32) float32 {
	a0 := archsimd.LoadFloat32x8Slice(a[0:8])
	a1 := archsimd.LoadFloat32x8Slice(a[8:16])
	b0 := archsimd.LoadFloat32x8Slice(b[0:8])
	b1 := archsimd.LoadFloat32x8Slice(b[8:16])
	d0 := a0.Sub(b0)
	d1 := a1.Sub(b1)
	sq := d1.Mul(d1).Add(d0.Mul(d0))
	var out [8]float32
	sq.StoreSlice(out[:])
	return out[0] + out[1] + out[2] + out[3] + out[4] + out[5] + out[6] + out[7]
}

// Centroids is K rows of 16 floats (lanes 14,15 = 0).
type Centroids [NClusters][16]float32

// KMeans runs Lloyd's algorithm with deterministic evenly-spaced init, writing
// the final assignment of every ref into assignments (len n). The assign phase
// is parallelized across cores (this is an offline tool).
func KMeans(refs []Ref, iters int) (cent *Centroids, assignments []int32) {
	n := len(refs)
	cent = new(Centroids)
	assignments = make([]int32, n)

	step := n / NClusters
	if step < 1 {
		step = 1
	}
	for c := 0; c < NClusters; c++ {
		src := c * step
		if src >= n {
			src = n - 1
		}
		copy(cent[c][0:NDims], refs[src].V[0:NDims]) // lanes 14,15 remain 0
	}

	workers := runtime.NumCPU()
	for it := 0; it < iters; it++ {
		// Phase A: assign each ref to its nearest centroid (parallel).
		var wg sync.WaitGroup
		chunk := (n + workers - 1) / workers
		for w := 0; w < workers; w++ {
			lo := w * chunk
			hi := lo + chunk
			if hi > n {
				hi = n
			}
			if lo >= hi {
				break
			}
			wg.Add(1)
			go func(lo, hi int) {
				defer wg.Done()
				for i := lo; i < hi; i++ {
					best := l2sqF32(&refs[i].V, &cent[0])
					bestC := int32(0)
					for c := 1; c < NClusters; c++ {
						if d := l2sqF32(&refs[i].V, &cent[c]); d < best {
							best = d
							bestC = int32(c)
						}
					}
					assignments[i] = bestC
				}
			}(lo, hi)
		}
		wg.Wait()

		// Phase B: recompute centroids (double accumulation, then divide).
		var sums [NClusters][NDims]float64
		var counts [NClusters]uint64
		for i := 0; i < n; i++ {
			c := assignments[i]
			counts[c]++
			row := &sums[c]
			v := &refs[i].V
			for d := 0; d < NDims; d++ {
				row[d] += float64(v[d])
			}
		}
		for c := 0; c < NClusters; c++ {
			if counts[c] == 0 {
				continue
			}
			inv := 1.0 / float64(counts[c])
			for d := 0; d < NDims; d++ {
				cent[c][d] = float32(sums[c][d] * inv)
			}
		}
	}
	return cent, assignments
}

// CountingSortByCluster returns the prefix-sum offsets (len K+1) and the order
// slice (len n): order[pos] is the original ref index landing at position pos.
func CountingSortByCluster(assignments []int32) (offsets []uint32, order []uint32) {
	n := len(assignments)
	offsets = make([]uint32, NClusters+1)
	order = make([]uint32, n)
	for _, c := range assignments {
		offsets[c+1]++
	}
	for c := 0; c < NClusters; c++ {
		offsets[c+1] += offsets[c]
	}
	cursor := make([]uint32, NClusters)
	copy(cursor, offsets[:NClusters])
	for i, c := range assignments {
		order[cursor[c]] = uint32(i)
		cursor[c]++
	}
	return offsets, order
}

// BBoxPack walks the cluster-ordered refs, computing per-cluster int16 bounding
// boxes and striping the quantized dims into the 7 SoA pair arrays.
func BBoxPack(refs []Ref, order, offsets []uint32) (bboxMin, bboxMax []int16, pairArr [NPairs][]int32, labels []uint8) {
	n := len(order)
	bboxMin = make([]int16, NClusters*16)
	bboxMax = make([]int16, NClusters*16)
	for c := 0; c < NClusters; c++ {
		for lane := 0; lane < NDims; lane++ {
			bboxMin[c*16+lane] = math.MaxInt16
			bboxMax[c*16+lane] = math.MinInt16
		}
		// lanes 14,15 stay 0
	}
	for p := 0; p < NPairs; p++ {
		pairArr[p] = make([]int32, n)
	}
	labels = make([]uint8, n)

	cid := 0
	for pos := 0; pos < n; pos++ {
		for offsets[cid+1] <= uint32(pos) {
			cid++
		}
		ref := &refs[order[pos]]
		labels[pos] = ref.Label
		qv := quantize(&ref.V)

		base := cid * 16
		for lane := 0; lane < 16; lane++ {
			if qv[lane] < bboxMin[base+lane] {
				bboxMin[base+lane] = qv[lane]
			}
			if qv[lane] > bboxMax[base+lane] {
				bboxMax[base+lane] = qv[lane]
			}
		}
		for p := 0; p < NPairs; p++ {
			lo := uint32(uint16(qv[2*p]))
			hi := uint32(uint16(qv[2*p+1]))
			pairArr[p][pos] = int32(lo | hi<<16)
		}
	}
	return bboxMin, bboxMax, pairArr, labels
}

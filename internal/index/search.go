package index

import "simd/archsimd"

// HasAVX2 reports whether the CPU supports the AVX2 instructions the search
// kernels rely on.
func HasAVX2() bool { return archsimd.X86.AVX2() }

// computeClusterPacked fills out[c] = (lb_c << CidBits) | c for every cluster,
// where lb_c is the squared bbox lower bound of the query against cluster c.
// Processes 8 clusters per SIMD iteration over the pair SoA.
func (ix *IvfIndex) computeClusterPacked(qp *[NPairs]archsimd.Int16x16, out []int64) {
	nGroups := (ix.NClusters + 7) / 8
	var zero16 [16]int16
	z := archsimd.LoadInt16x16(&zero16)
	var sq [8]int32

	for g := 0; g < nGroups; g++ {
		base := g * NPairs * 16
		// pair 0 seeds the accumulator
		bmin := archsimd.LoadInt16x16Slice(ix.bpsoaMin[base:])
		bmax := archsimd.LoadInt16x16Slice(ix.bpsoaMax[base:])
		below := bmin.Sub(qp[0]).Max(z) // max(bmin - q, 0)
		above := qp[0].Sub(bmax).Max(z) // max(q - bmax, 0)
		gap := below.Max(above)
		acc := gap.DotProductPairs(gap) // Int32x8: one squared gap per cluster

		for p := 1; p < NPairs; p++ {
			o := base + p*16
			bmn := archsimd.LoadInt16x16Slice(ix.bpsoaMin[o:])
			bmx := archsimd.LoadInt16x16Slice(ix.bpsoaMax[o:])
			bl := bmn.Sub(qp[p]).Max(z)
			ab := qp[p].Sub(bmx).Max(z)
			gp := bl.Max(ab)
			acc = acc.Add(gp.DotProductPairs(gp))
		}
		acc.StoreSlice(sq[:])

		for l := 0; l < 8; l++ {
			c := g*8 + l
			if c >= ix.NClusters {
				break
			}
			lb := int64(sq[l]) // sign-extend, matching the asm widening
			out[c] = (lb << CidBits) | int64(c)
		}
	}
}

// pairSq returns the per-vector squared diff of pair p across 8 vectors.
func (ix *IvfIndex) pairSq(p, off int, qp *[NPairs]archsimd.Int16x16) archsimd.Int32x8 {
	d := archsimd.LoadInt16x16Slice(ix.pairs[p][off:]).Sub(qp[p])
	return d.DotProductPairs(d)
}

// scanCluster computes the exact L2 distance of every vector in cluster bestC
// against the query and folds it into the 5-entry top-K (packed key
// (dist<<IdxBits)|idx), keeping worstKey = max(top-K) current.
//
// Early-termination gate: the 7 pairs are summed in three stages; after the
// first two, if every vector in the batch already exceeds worst_dist the whole
// batch is skipped (no vector can enter the top-K). Result is identical to the
// full sum — only batches that can't win are short-circuited.
func (ix *IvfIndex) scanCluster(bestC int, qp *[NPairs]archsimd.Int16x16, topkK *[5]int64, topkL *[5]uint8, worstKey *int64) {
	start := int(ix.clusterOffsets[bestC])
	end := int(ix.clusterOffsets[bestC+1])
	count := end - start
	wk := *worstKey
	var dists [8]int32

	for i := 0; i < count; i += 8 {
		base := start + i
		off := 2 * base

		// gate threshold from the current worst; inactive until top-K is full
		// (worst_dist > INT32_MAX). Overflow of a partial only ever fails to
		// prune (safe), never prunes a vector that could win.
		wd := wk >> IdxBits
		gate := wd <= 0x7FFFFFFF
		var thresh archsimd.Int32x8
		if gate {
			thresh = archsimd.BroadcastInt32x8(int32(wd))
		}

		// Two independent accumulator chains (A={3,0,2,6}, B={5,1,4}) break the
		// serial add dependency so the squared-pair sums pipeline, fused only at
		// the gate checks and the end. Pair order {3,5},{0,1},{2,4,6} also
		// front-loads the high-variance dims so the gate prunes as early as
		// possible.
		// stage 1: A=pair3, B=pair5
		accA := ix.pairSq(3, off, qp)
		accB := ix.pairSq(5, off, qp)
		if gate && accA.Add(accB).Less(thresh).ToBits() == 0 {
			continue
		}
		// stage 2: A+=pair0, B+=pair1
		accA = accA.Add(ix.pairSq(0, off, qp))
		accB = accB.Add(ix.pairSq(1, off, qp))
		if gate && accA.Add(accB).Less(thresh).ToBits() == 0 {
			continue
		}
		// stage 3: A+=pair2+pair6, B+=pair4 → full L2 = A+B
		accA = accA.Add(ix.pairSq(2, off, qp)).Add(ix.pairSq(6, off, qp))
		accB = accB.Add(ix.pairSq(4, off, qp))
		accA.Add(accB).StoreSlice(dists[:])

		valid := count - i
		if valid > 8 {
			valid = 8
		}
		for j := 0; j < valid; j++ {
			key := (int64(uint32(dists[j])) << IdxBits) | int64(base+j)
			if key >= wk {
				continue
			}
			// replace the current worst slot
			wi, mx := 0, topkK[0]
			for t := 1; t < 5; t++ {
				if topkK[t] > mx {
					mx, wi = topkK[t], t
				}
			}
			topkK[wi] = key
			topkL[wi] = ix.labels[base+j]
			// recompute worst
			wk = topkK[0]
			for t := 1; t < 5; t++ {
				if topkK[t] > wk {
					wk = topkK[t]
				}
			}
		}
	}
	*worstKey = wk
}

// searchCore runs phases 1-3 plus the repair pattern, leaving the top-K in
// topkK/topkL. The query is a 16-lane int16 vector (dims 14,15 = 0).
func (ix *IvfIndex) searchCore(q *[16]int16, maxProbes int, topkK *[5]int64, topkL *[5]uint8) {
	const maxI64 = int64(0x7FFFFFFFFFFFFFFF)

	// qpair[p] broadcasts the (q[2p], q[2p+1]) pair across all 8 lane slots.
	var qpairs [NPairs][16]int16
	for p := 0; p < NPairs; p++ {
		lo, hi := q[2*p], q[2*p+1]
		for l := 0; l < 8; l++ {
			qpairs[p][2*l] = lo
			qpairs[p][2*l+1] = hi
		}
	}
	var qp [NPairs]archsimd.Int16x16
	for p := 0; p < NPairs; p++ {
		qp[p] = archsimd.LoadInt16x16(&qpairs[p])
	}

	var packed [NClusters]int64
	ix.computeClusterPacked(&qp, packed[:])

	for i := 0; i < 5; i++ {
		topkK[i] = maxI64
	}
	*topkL = [5]uint8{}
	worstKey := maxI64

	probe := 0
	repairDone := false
	for {
	probeLoop:
		for probe < maxProbes {
			// pick the cluster with the smallest packed key
			best := maxI64
			for c := 0; c < ix.NClusters; c++ {
				if packed[c] < best {
					best = packed[c]
				}
			}
			if best == maxI64 {
				break probeLoop // all tombstoned
			}
			bestLb := best >> CidBits
			if (bestLb << IdxBits) >= worstKey {
				break probeLoop // no remaining cluster can improve top-K
			}
			bestC := int(best & CidMask)
			packed[bestC] = maxI64 // tombstone
			ix.scanCluster(bestC, &qp, topkK, topkL, &worstKey)
			probe++
		}

		// Repair: the fraud verdict is the top-5 label count alone. If it is
		// unambiguous (0 = all legit, 5 = all fraud) we're done; if ambiguous
		// ([1,4]) extend the budget to a full sweep (lb-prune still cuts it).
		if repairDone {
			return
		}
		if worstKey != maxI64 {
			cnt := 0
			for _, l := range topkL {
				cnt += int(l)
			}
			if cnt < NProbeRepairMin || cnt > NProbeRepairMax {
				return
			}
		}
		repairDone = true
		maxProbes = ix.NClusters
	}
}

// Search returns the fraud count (0..5) among the query's 5 nearest neighbours.
func (ix *IvfIndex) Search(q *[16]int16) uint8 {
	var topkK [5]int64
	var topkL [5]uint8
	ix.searchCore(q, NProbeInitial, &topkK, &topkL)
	cnt := uint8(0)
	for _, l := range topkL {
		cnt += l
	}
	return cnt
}

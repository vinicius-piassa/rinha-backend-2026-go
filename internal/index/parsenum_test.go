package index

import (
	"math/rand"
	"strconv"
	"testing"
)

// TestParseNumberMatchesStrconv proves the fast decimal path is bit-identical
// to strconv.ParseFloat across corpus-like values and random fixtures, so the
// generated index is unchanged.
func TestParseNumberMatchesStrconv(t *testing.T) {
	fixed := []string{
		"0", "0.0", "1", "1.0", "-1", "-1.0", "384.88", "0.0833", "0.05",
		"0.9565", "0.4993", "13.7", "18.8", "298.95", "769.76", "1000",
		"0.002", "0.85", "0.15", "-0.5", "9500", "0.00001", "12345.6789",
	}
	check := func(s string) {
		want, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return
		}
		b := []byte(s)
		got, np, ok := parseNumber(b, 0)
		if !ok || np != len(b) {
			t.Fatalf("parseNumber(%q) ok=%v np=%d/%d", s, ok, np, len(b))
		}
		if got != want {
			t.Errorf("parseNumber(%q)=%v want %v (bits %x vs %x)", s, got, want, got, want)
		}
	}
	for _, s := range fixed {
		check(s)
	}
	// random decimals with 0..6 fractional digits
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 200000; i++ {
		ip := rng.Intn(100000)
		fd := rng.Intn(7)
		var s string
		if fd == 0 {
			s = strconv.Itoa(ip)
		} else {
			s = strconv.Itoa(ip) + "." + leftPad(rng.Intn(intPow(10, fd)), fd)
		}
		if rng.Intn(2) == 0 {
			s = "-" + s
		}
		check(s)
	}
}

func intPow(b, e int) int {
	r := 1
	for i := 0; i < e; i++ {
		r *= b
	}
	return r
}

func leftPad(v, width int) string {
	s := strconv.Itoa(v)
	for len(s) < width {
		s = "0" + s
	}
	return s
}

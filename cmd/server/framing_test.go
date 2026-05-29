package main

import (
	"bytes"
	"testing"
)

// BenchmarkFraming exercises the per-request userspace path (header framing +
// content-length scan + response selection) and reports allocations. The hot
// path must be 0 allocs/op so the GC can stay disabled.
func BenchmarkFraming(b *testing.B) {
	req := []byte("POST /fraud-score HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: 7\r\n\r\n{\"a\":1}")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		he := bytes.Index(req, hdrSep)
		if he < 0 {
			b.Fatal("no header sep")
		}
		if cl := contentLength(req[:he+4]); cl != 7 {
			b.Fatalf("content-length = %d, want 7", cl)
		}
	}
}

func TestContentLengthFold(t *testing.T) {
	cases := map[string]int{
		"Content-Length: 42\r\n\r\n": 42,
		"content-length:7\r\n\r\n":   7,
		"CONTENT-LENGTH:   100\r\n":  100,
		"X-Other: 1\r\n\r\n":         -1,
	}
	for hdr, want := range cases {
		if got := contentLength([]byte(hdr)); got != want {
			t.Errorf("contentLength(%q) = %d, want %d", hdr, got, want)
		}
	}
}

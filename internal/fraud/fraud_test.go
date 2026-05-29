package fraud

import (
	"testing"
	"time"
)

var warmBody = []byte(`{"id":"tx-warm","transaction":{"amount":384.88,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},"customer":{"avg_amount":769.76,"tx_count_24h":3,"known_merchants":["MERC-009","MERC-001"]},"merchant":{"id":"MERC-001","mcc":"5912","avg_amount":298.95},"terminal":{"is_online":false,"card_present":true,"km_from_home":13.7},"last_transaction":{"timestamp":"2026-03-11T14:58:35Z","km_from_current":18.8}}`)

func TestParseWarmBody(t *testing.T) {
	var r Request
	if !ParseRequest(warmBody, &r) {
		t.Fatal("ParseRequest failed")
	}
	wantTS := time.Date(2026, 3, 11, 20, 23, 35, 0, time.UTC).Unix()
	wantLast := time.Date(2026, 3, 11, 14, 58, 35, 0, time.UTC).Unix()

	checks := []struct {
		name string
		got  any
		want any
	}{
		{"Amount", r.Amount, 384.88},
		{"Installments", r.Installments, int32(3)},
		{"TS", r.TS, wantTS},
		{"CustomerAvg", r.CustomerAvg, 769.76},
		{"TxCount24h", r.TxCount24h, int32(3)},
		{"MCC", string(r.MCC[:]), "5912"},
		{"MerchantAvg", r.MerchantAvg, 298.95},
		{"IsOnline", r.IsOnline, false},
		{"CardPresent", r.CardPresent, true},
		{"KmHome", r.KmHome, 13.7},
		{"HasLastTx", r.HasLastTx, true},
		{"LastTS", r.LastTS, wantLast},
		{"KmLast", r.KmLast, 18.8},
		{"KnownMerchant", r.KnownMerchant, true},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
}

func TestVectorizeTag(t *testing.T) {
	var r Request
	if !ParseRequest(warmBody, &r) {
		t.Fatal("parse failed")
	}
	v := Vectorize(&r)
	// known merchant → unknown=0; has_last_tx → 1  ⇒ tag = 1
	if got := Tag(&v); got != 1 {
		t.Errorf("tag = %d, want 1 (v[11]=%d v[5]=%d)", got, v[11], v[5])
	}
	t.Logf("query = %v", v[:14])
}

func BenchmarkParseVectorize(b *testing.B) {
	var r Request
	b.ReportAllocs()
	b.ResetTimer()
	var sink int16
	for i := 0; i < b.N; i++ {
		ParseRequest(warmBody, &r)
		v := Vectorize(&r)
		sink += v[0]
	}
	_ = sink
}

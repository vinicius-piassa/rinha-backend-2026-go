package index

import (
	"errors"
	"strconv"
	"unsafe"
)

// ErrParse is returned on any structural problem in the corpus JSON.
var ErrParse = errors.New("index: corpus JSON parse error")

func isWs(c byte) bool { return c == ' ' || c == '\t' || c == '\n' || c == '\r' }

// pow10 holds exact float64 powers of ten for the fast-path divisor.
var pow10 = [...]float64{
	1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
	1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18,
}

// parseNumber reads a JSON number token at p. For the common sign/int/frac
// decimals in the corpus it accumulates the full integer mantissa and divides
// by a power of ten — a single rounding identical to strconv's atof64exact —
// skipping strconv's general scanner. Falls back to strconv on exponents or
// mantissa overflow.
func parseNumber(buf []byte, p int) (float64, int, bool) {
	start := p
	n := len(buf)
	neg := false
	if p < n && (buf[p] == '-' || buf[p] == '+') {
		neg = buf[p] == '-'
		p++
	}
	var mant uint64
	digits, fracDigits := 0, 0
	for p < n && buf[p] >= '0' && buf[p] <= '9' {
		mant = mant*10 + uint64(buf[p]-'0')
		p++
		digits++
	}
	if p < n && buf[p] == '.' {
		p++
		for p < n && buf[p] >= '0' && buf[p] <= '9' {
			mant = mant*10 + uint64(buf[p]-'0')
			p++
			digits++
			fracDigits++
		}
	}
	// exponent or too many digits → general path for exactness/safety
	if (p < n && (buf[p] == 'e' || buf[p] == 'E')) || digits > 18 {
		for p < n {
			c := buf[p]
			if (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.' || c == 'e' || c == 'E' {
				p++
				continue
			}
			break
		}
		f, err := strconv.ParseFloat(unsafe.String(&buf[start], p-start), 64)
		if err != nil {
			return 0, p, false
		}
		return f, p, true
	}
	if digits == 0 {
		return 0, start, false
	}
	val := float64(mant)
	if fracDigits > 0 {
		val /= pow10[fracDigits]
	}
	if neg {
		val = -val
	}
	return val, p, true
}

// skipValue advances past one JSON value (used for unknown keys).
func skipValue(buf []byte, p int) (int, bool) {
	n := len(buf)
	for p < n && isWs(buf[p]) {
		p++
	}
	if p >= n {
		return p, false
	}
	switch buf[p] {
	case '"':
		p++
		for p < n && buf[p] != '"' {
			if buf[p] == '\\' {
				p++
			}
			p++
		}
		if p >= n {
			return p, false
		}
		return p + 1, true
	case '{', '[':
		open, close := buf[p], byte(']')
		if open == '{' {
			close = '}'
		}
		depth := 0
		for p < n {
			c := buf[p]
			switch c {
			case '"':
				np, ok := skipValue(buf, p)
				if !ok {
					return p, false
				}
				p = np
				continue
			case open:
				depth++
			case close:
				depth--
				if depth == 0 {
					return p + 1, true
				}
			}
			p++
		}
		return p, false
	default: // number, true, false, null
		for p < n && buf[p] != ',' && buf[p] != '}' && buf[p] != ']' && !isWs(buf[p]) {
			p++
		}
		return p, true
	}
}

// ParseRefs parses the corpus array of {"vector":[14 floats],"label":"…"} into
// Ref values. Label is 1 iff exactly "fraud".
func ParseRefs(buf []byte, maxRefs int) ([]Ref, error) {
	refs := make([]Ref, 0, maxRefs) // preallocate to avoid append regrowth/copy
	p, n := 0, len(buf)
	skipWs := func() {
		for p < n && isWs(buf[p]) {
			p++
		}
	}

	skipWs()
	if p >= n || buf[p] != '[' {
		return nil, ErrParse
	}
	p++

	for {
		skipWs()
		if p >= n {
			return refs, nil
		}
		switch buf[p] {
		case ']':
			return refs, nil
		case ',':
			p++
			continue
		case '{':
			p++
		default:
			return nil, ErrParse
		}
		if len(refs) >= maxRefs {
			return nil, ErrParse
		}

		var ref Ref
		for {
			skipWs()
			if p >= n {
				return nil, ErrParse
			}
			if buf[p] == '}' {
				p++
				break
			}
			if buf[p] == ',' {
				p++
				continue
			}
			if buf[p] != '"' {
				return nil, ErrParse
			}
			p++
			ks := p
			for p < n && buf[p] != '"' {
				p++
			}
			if p >= n {
				return nil, ErrParse
			}
			key := buf[ks:p]
			p++
			skipWs()
			if p >= n || buf[p] != ':' {
				return nil, ErrParse
			}
			p++
			skipWs()

			switch {
			case len(key) == 6 && string(key) == "vector":
				if p >= n || buf[p] != '[' {
					return nil, ErrParse
				}
				p++
				for dim := 0; dim < NDims; dim++ {
					skipWs()
					f, np, ok := parseNumber(buf, p)
					if !ok {
						return nil, ErrParse
					}
					p = np
					ref.V[dim] = float32(f)
					skipWs()
					if p < n && buf[p] == ',' {
						p++
					}
				}
				skipWs()
				if p >= n || buf[p] != ']' {
					return nil, ErrParse
				}
				p++
			case len(key) == 5 && string(key) == "label":
				if p >= n || buf[p] != '"' {
					return nil, ErrParse
				}
				p++
				ls := p
				for p < n && buf[p] != '"' {
					p++
				}
				if p >= n {
					return nil, ErrParse
				}
				if string(buf[ls:p]) == "fraud" {
					ref.Label = 1
				}
				p++
			default:
				np, ok := skipValue(buf, p)
				if !ok {
					return nil, ErrParse
				}
				p = np
			}
		}
		refs = append(refs, ref)
	}
}

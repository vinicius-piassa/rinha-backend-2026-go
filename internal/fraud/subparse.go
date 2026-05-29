package fraud

// expectObjOpen consumes leading whitespace and the opening '{'.
func (s *psr) expectObjOpen() bool {
	s.ws()
	if s.p >= s.end || s.b[s.p] != '{' {
		return false
	}
	s.p++
	return true
}

// nextKey reads the next "key": inside an object, positioning at the value.
// more=false means the closing '}' was reached.
func (s *psr) nextKey() (key []byte, more, ok bool) {
	s.ws()
	if s.p >= s.end {
		return nil, false, false
	}
	if s.b[s.p] == '}' {
		s.p++
		return nil, false, true
	}
	cs, ce, o := s.skipString()
	if !o {
		return nil, false, false
	}
	s.ws()
	if s.p >= s.end || s.b[s.p] != ':' {
		return nil, false, false
	}
	s.p++
	s.ws()
	return s.b[cs:ce], true, true
}

func (s *psr) parseTransaction(r *Request) bool {
	if !s.expectObjOpen() {
		return false
	}
	for {
		key, more, ok := s.nextKey()
		if !ok {
			return false
		}
		if !more {
			return true
		}
		switch string(key) {
		case "amount":
			v, o := s.number()
			if !o {
				return false
			}
			r.Amount = v
		case "installments":
			v, o := s.int32v()
			if !o {
				return false
			}
			r.Installments = v
		case "requested_at":
			cs, ce, o := s.skipString()
			if !o {
				return false
			}
			r.TS = parseISO8601(s.b[cs:ce])
		default:
			if !s.skipValue() {
				return false
			}
		}
		s.afterValue()
	}
}

func (s *psr) parseCustomer(r *Request) bool {
	if !s.expectObjOpen() {
		return false
	}
	for {
		key, more, ok := s.nextKey()
		if !ok {
			return false
		}
		if !more {
			return true
		}
		switch string(key) {
		case "avg_amount":
			v, o := s.number()
			if !o {
				return false
			}
			r.CustomerAvg = v
		case "tx_count_24h":
			v, o := s.int32v()
			if !o {
				return false
			}
			r.TxCount24h = v
		case "known_merchants":
			s.ws()
			if s.p >= s.end || s.b[s.p] != '[' {
				return false
			}
			start := s.p
			if !s.skipValue() {
				return false
			}
			s.kmStart, s.kmEnd = start, s.p
		default:
			if !s.skipValue() {
				return false
			}
		}
		s.afterValue()
	}
}

func (s *psr) parseMerchant(r *Request) bool {
	if !s.expectObjOpen() {
		return false
	}
	gotMcc := false
	for {
		key, more, ok := s.nextKey()
		if !ok {
			return false
		}
		if !more {
			return gotMcc // mcc is required
		}
		switch string(key) {
		case "id":
			cs, ce, o := s.skipString()
			if !o {
				return false
			}
			s.midStart, s.midLen = cs, ce-cs
		case "mcc":
			cs, ce, o := s.skipString()
			if !o {
				return false
			}
			n := ce - cs
			for i := 0; i < 4; i++ {
				if i < n {
					r.MCC[i] = s.b[cs+i]
				} else {
					r.MCC[i] = '0'
				}
			}
			gotMcc = true
		case "avg_amount":
			v, o := s.number()
			if !o {
				return false
			}
			r.MerchantAvg = v
		default:
			if !s.skipValue() {
				return false
			}
		}
		s.afterValue()
	}
}

func (s *psr) parseTerminal(r *Request) bool {
	if !s.expectObjOpen() {
		return false
	}
	for {
		key, more, ok := s.nextKey()
		if !ok {
			return false
		}
		if !more {
			return true
		}
		switch string(key) {
		case "is_online":
			r.IsOnline = s.p < s.end && s.b[s.p] == 't'
			if !s.skipValue() {
				return false
			}
		case "card_present":
			r.CardPresent = s.p < s.end && s.b[s.p] == 't'
			if !s.skipValue() {
				return false
			}
		case "km_from_home":
			v, o := s.number()
			if !o {
				return false
			}
			r.KmHome = v
		default:
			if !s.skipValue() {
				return false
			}
		}
		s.afterValue()
	}
}

func (s *psr) parseLastTx(r *Request) bool {
	s.ws()
	// null → no last transaction
	if s.p+4 <= s.end && string(s.b[s.p:s.p+4]) == "null" {
		r.HasLastTx = false
		s.p += 4
		return true
	}
	if !s.expectObjOpen() {
		return false
	}
	r.HasLastTx = true
	for {
		key, more, ok := s.nextKey()
		if !ok {
			return false
		}
		if !more {
			return true
		}
		switch string(key) {
		case "timestamp":
			cs, ce, o := s.skipString()
			if !o {
				return false
			}
			r.LastTS = parseISO8601(s.b[cs:ce])
		case "km_from_current":
			v, o := s.number()
			if !o {
				return false
			}
			r.KmLast = v
		default:
			if !s.skipValue() {
				return false
			}
		}
		s.afterValue()
	}
}

// parseISO8601 converts "YYYY-MM-DDTHH:MM:SS[.fff][Z|±HH:MM]" to Unix seconds
// via days_from_civil. Returns 0 on length/separator validation failure.
func parseISO8601(s []byte) int64 {
	if len(s) < 19 {
		return 0
	}
	if s[4] != '-' || s[7] != '-' {
		return 0
	}
	if s[10] != 'T' && s[10] != ' ' {
		return 0
	}
	if s[13] != ':' || s[16] != ':' {
		return 0
	}
	d2 := func(i int) int { return int(s[i]-'0')*10 + int(s[i+1]-'0') }

	year := int(s[0]-'0')*1000 + int(s[1]-'0')*100 + int(s[2]-'0')*10 + int(s[3]-'0')
	month := d2(5)
	day := d2(8)
	hour := d2(11)
	mn := d2(14)
	sec := d2(17)

	y := year
	if month <= 2 {
		y--
	}
	var era int
	if y >= 0 {
		era = y / 400
	} else {
		era = (y - 399) / 400
	}
	yoe := y - era*400
	var m int
	if month > 2 {
		m = month - 3
	} else {
		m = month + 9
	}
	doy := (153*m+2)/5 + day - 1
	doe := yoe*365 + yoe/4 - yoe/100 + doy
	days := int64(era)*146097 + int64(doe) - 719468
	epoch := days*86400 + int64(hour)*3600 + int64(mn)*60 + int64(sec)

	// optional fractional seconds + timezone
	i := 19
	if i < len(s) && s[i] == '.' {
		i++
		for i < len(s) && s[i] >= '0' && s[i] <= '9' {
			i++
		}
	}
	if i < len(s) {
		c := s[i]
		if (c == '+' || c == '-') && len(s)-i >= 6 && s[i+3] == ':' {
			off := int64(d2(i+1))*3600 + int64(d2(i+4))*60
			if c == '+' {
				epoch -= off
			} else {
				epoch += off
			}
		}
	}
	return epoch
}

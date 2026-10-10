package service

import (
	"strings"
	"unicode"
)

// charMap is the per-character normalization from ml/text.py (_NORMALIZE).
var charMap = map[rune]rune{
	'ё': 'е', 'Ё': 'Е', '«': '"', '»': '"', '„': '"', '“': '"', '”': '"',
	'×': 'x', '\t': ' ', '—': '-', '–': '-',
	0xa0: ' ', // NBSP
}

// normalize is a port of ml.text.normalize: rune count is preserved.
func normalize(s string) string {
	r := []rune(s)
	for i, c := range r {
		if m, ok := charMap[c]; ok {
			r[i] = m
		}
	}
	// «х/б» -> «хб » as a whole word; Go's \b is ASCII-only, so boundaries are checked by hand.
	for i := 0; i+2 < len(r); i++ {
		if (r[i] == 'х' || r[i] == 'Х') && r[i+1] == '/' && (r[i+2] == 'б' || r[i+2] == 'Б') &&
			(i == 0 || !isWordRune(r[i-1])) && (i+3 == len(r) || !isWordRune(r[i+3])) {
			r[i+1], r[i+2] = 'б', ' '
		}
	}
	return string(r)
}

// isWordRune matches Python's \w (str.isalnum or "_"): numbers include ², ½, Ⅻ, not only 0-9.
func isWordRune(c rune) bool {
	return c == '_' || unicode.IsLetter(c) || unicode.IsNumber(c)
}

// preprocess cleans the user text before it is sent to the model service:
// ml normalization, then invisible control/format characters are dropped and whitespace
// runs (including newlines) are collapsed to a single space.
func preprocess(s string) string {
	s = strings.Map(func(c rune) rune {
		if !unicode.IsSpace(c) && (unicode.IsControl(c) || unicode.Is(unicode.Cf, c)) {
			return -1
		}
		return c
	}, normalize(s))
	return strings.Join(strings.Fields(s), " ")
}

package service

import (
	"testing"
	"unicode/utf8"
)

func TestNormalize(t *testing.T) {
	cases := map[string]string{
		"ёлка Ёж":                  "елка Еж",
		"«кресло» „x“ ”y”":         `"кресло" "x" "y"`,
		"100×50":                   "100x50",
		"а\xc2\xa0б\tв":            "а б в", // NBSP, tab
		"1—2–3":                    "1-2-3",
		"футболка х/б, Х/Б":        "футболка хб , Хб ",
		"х/б":                      "хб ",
		"тх/б х/бх 1х/б х/б2 _х/б": "тх/б х/бх 1х/б х/б2 _х/б", // not whole words
		"(х/б)":                    "(хб )",
		// boundaries as in Python: ², ½, Ⅻ are word characters, a combining mark is not
		"х/б\xc2\xb2":     "х/б\xc2\xb2",
		"\xc2\xbdх/б":     "\xc2\xbdх/б",
		"х/б\xe2\x85\xab": "х/б\xe2\x85\xab",
		"х/б\xcc\x81":     "хб \xcc\x81",
	}
	for in, want := range cases {
		got := normalize(in)
		if got != want {
			t.Errorf("normalize(%q) = %q, want %q", in, got, want)
		}
		if utf8.RuneCountInString(got) != utf8.RuneCountInString(in) {
			t.Errorf("normalize(%q) changed length", in)
		}
	}
}

func TestPreprocess(t *testing.T) {
	cases := map[string]string{
		"  стул\n\nофисный\t«Престиж» ":        `стул офисный "Престиж"`,
		"футболка х/б синяя":                   "футболка хб синяя",
		"сту\xe2\x80\x8bл\xc2\xad офисный\x00": "стул офисный", // zero-width space, soft hyphen, NUL
		"\n\t ": "",
	}
	for in, want := range cases {
		if got := preprocess(in); got != want {
			t.Errorf("preprocess(%q) = %q, want %q", in, got, want)
		}
	}
}

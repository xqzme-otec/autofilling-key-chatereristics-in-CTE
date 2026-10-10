package sheet_test

import (
	"bytes"
	"errors"
	"os"
	"reflect"
	"testing"

	"github.com/xuri/excelize/v2"
	"golang.org/x/text/encoding/charmap"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/domain"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/sheet"
)

func TestReadCSV(t *testing.T) {
	cp1251, err := charmap.Windows1251.NewEncoder().String("Название;Описание\r\nСтул;на колесиках, кожзам\r\n")
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name, data string
		want       [][]string
	}{
		{"semicolons, BOM", "\xef\xbb\xbfНазвание;Описание\nСтул;на колесиках, кожзам\n",
			[][]string{{"Название", "Описание"}, {"Стул", "на колесиках, кожзам"}}},
		{"windows-1251 from Excel", cp1251,
			[][]string{{"Название", "Описание"}, {"Стул", "на колесиках, кожзам"}}},
		{"commas, quoted description", "name,description\nСтул,\"на колесиках, кожзам\"\n",
			[][]string{{"name", "description"}, {"Стул", "на колесиках, кожзам"}}},
		{"tabs", "Стул\tна колесиках; кожзам\n", [][]string{{"Стул", "на колесиках; кожзам"}}},
		{"empty lines keep row numbers", "Стул;офисный\n\n\nСтол;письменный\n\n",
			[][]string{{"Стул", "офисный"}, nil, nil, {"Стол", "письменный"}}},
		{"stray quote", "Стул;экран 15\" матовый\n", [][]string{{"Стул", "экран 15\" матовый"}}},
		{"empty file", "", nil},
	}
	for _, c := range cases {
		rows, err := sheet.Read("Товары.CSV", []byte(c.data))
		if err != nil || !reflect.DeepEqual(rows, c.want) {
			t.Errorf("%s: got %q, %v; want %q", c.name, rows, err, c.want)
		}
	}
}

func TestReadXLSX(t *testing.T) {
	f := excelize.NewFile()
	first := f.GetSheetName(0)
	for ref, value := range map[string]any{"A1": "Название", "B1": "Описание", "A2": "Стул", "B2": "на колесиках",
		"B4": "молоток слесарный", "A5": "Стол", "B5": 120} {
		if err := f.SetCellValue(first, ref, value); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := f.NewSheet("Другой лист"); err != nil {
		t.Fatal(err)
	}
	f.SetCellValue("Другой лист", "A1", "не читается")
	var buf bytes.Buffer
	if err := f.Write(&buf); err != nil {
		t.Fatal(err)
	}

	rows, err := sheet.Read("products.xlsx", buf.Bytes())
	want := [][]string{{"Название", "Описание"}, {"Стул", "на колесиках"}, nil, {"", "молоток слесарный"}, {"Стол", "120"}}
	if err != nil || !reflect.DeepEqual(rows, want) {
		t.Errorf("got %q, %v; want %q", rows, err, want)
	}
}

// testdata/products.xls is a real BIFF8 workbook written by Python's xlwt: two sheets, an empty
// row, cells left blank and a numeric cell.
func TestReadXLS(t *testing.T) {
	data, err := os.ReadFile("testdata/products.xls")
	if err != nil {
		t.Fatal(err)
	}
	rows, err := sheet.Read("products.xls", data)
	want := [][]string{{"Название", "Описание"}, {"Стул офисный", "на колесиках, кожзам, высота 110 см"}, {""},
		{"", "молоток слесарный, ручка 25 см"}, {"Халат детский"}, {"Стол", "120"}}
	if err != nil || !reflect.DeepEqual(rows, want) {
		t.Errorf("got %q, %v; want %q", rows, err, want)
	}
}

func TestReadRejectsUnreadableFiles(t *testing.T) {
	xls, err := os.ReadFile("testdata/products.xls")
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name, filename string
		data           []byte
	}{
		{"unsupported extension", "products.pdf", []byte("%PDF-1.7")},
		{"no extension", "products", []byte("Стул;офисный")},
		{"text named .xlsx", "products.xlsx", []byte("Стул;офисный")},
		{"text named .xls", "products.xls", []byte("Стул;офисный")},
		{"truncated .xls", "products.xls", xls[:len(xls)/2]},
		{"empty .xlsx", "products.xlsx", nil},
		{"empty .xls", "products.xls", nil},
	}
	for _, c := range cases {
		rows, err := sheet.Read(c.filename, c.data)
		var de *domain.Error
		if !errors.As(err, &de) || de.Code != domain.CodeInvalidFile {
			t.Errorf("%s: got %q, %v; want an invalid_file error", c.name, rows, err)
		}
	}
}

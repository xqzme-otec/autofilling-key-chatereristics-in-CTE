// Package sheet reads an uploaded table (.csv, .xlsx, .xls) as rows of text cells. Only the first
// worksheet of a workbook is read; what the rows mean is decided by the service.
package sheet

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/shakinm/xlsReader/xls"
	"github.com/xuri/excelize/v2"
	"golang.org/x/text/encoding/charmap"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/domain"
)

// maxUnzippedBytes bounds what an .xlsx may expand to: a small archive can hold gigabytes of XML.
// excelize spills worksheets above its XML limit to temporary files, so both limits are equal
// and a larger workbook is rejected instead.
const maxUnzippedBytes = 16 << 20

// Read returns the rows of the table; the format is chosen by the file name extension. Row i of
// the result is row i+1 of the file: empty rows are kept, trailing ones are dropped.
func Read(filename string, data []byte) ([][]string, error) {
	var read func([]byte) ([][]string, error)
	switch strings.ToLower(filepath.Ext(filename)) {
	case ".csv":
		read = readCSV
	case ".xlsx":
		read = readXLSX
	case ".xls":
		read = readXLS
	default:
		return nil, domain.Errorf(domain.CodeInvalidFile, "Формат файла не поддерживается. Загрузите .csv, .xlsx или .xls")
	}
	rows, err := read(data)
	if err != nil {
		return nil, domain.Wrap(domain.CodeInvalidFile, err, "Не удалось прочитать файл: он повреждён или не соответствует своему расширению")
	}
	return trimEmptyTail(rows), nil
}

func trimEmptyTail(rows [][]string) [][]string {
	for len(rows) > 0 && strings.TrimSpace(strings.Join(rows[len(rows)-1], "")) == "" {
		rows = rows[:len(rows)-1]
	}
	return rows
}

// readCSV accepts what Excel saves: UTF-8 with or without a BOM or Windows-1251 (the Russian
// locale default), with any of the usual delimiters.
func readCSV(data []byte) ([][]string, error) {
	data = bytes.TrimPrefix(data, []byte("\xef\xbb\xbf"))
	if !utf8.Valid(data) {
		decoded, err := charmap.Windows1251.NewDecoder().Bytes(data)
		if err != nil {
			return nil, err
		}
		data = decoded
	}
	r := csv.NewReader(bytes.NewReader(data))
	r.Comma = delimiter(data)
	r.FieldsPerRecord = -1 // rows may differ in length
	r.LazyQuotes = true    // a stray quote in a free-form description is not an error

	// csv.Reader skips empty lines; FieldPos restores the line of every record.
	var rows [][]string
	for {
		record, err := r.Read()
		if err == io.EOF {
			return rows, nil
		}
		if err != nil {
			return nil, err
		}
		line, _ := r.FieldPos(0)
		for len(rows) < line-1 {
			rows = append(rows, nil)
		}
		rows = append(rows, record)
	}
}

// delimiter picks the field separator by the first line. A tab or a semicolon wins over commas:
// they rarely occur in a product description, and commas occur in almost every one.
func delimiter(data []byte) rune {
	line, _, _ := bytes.Cut(data, []byte("\n"))
	for _, d := range "\t;" {
		if bytes.ContainsRune(line, d) {
			return d
		}
	}
	return ','
}

func readXLSX(data []byte) ([][]string, error) {
	f, err := excelize.OpenReader(bytes.NewReader(data),
		excelize.Options{UnzipSizeLimit: maxUnzippedBytes, UnzipXMLSizeLimit: maxUnzippedBytes})
	if err != nil {
		return nil, err
	}
	defer f.Close()
	it, err := f.Rows(f.GetSheetName(0))
	if err != nil {
		return nil, err
	}
	defer it.Close()
	var rows [][]string
	for it.Next() {
		cells, err := it.Columns()
		if err != nil {
			return nil, err
		}
		rows = append(rows, cells)
	}
	return rows, it.Error()
}

// readXLS reads the legacy binary format. The library is no longer maintained and indexes into
// the file without bounds checks, so a panic on a damaged file becomes an error.
func readXLS(data []byte) (rows [][]string, err error) {
	defer func() {
		if p := recover(); p != nil {
			rows, err = nil, fmt.Errorf("xls reader panic: %v", p)
		}
	}()
	wb, err := xls.OpenReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	sheets := wb.GetSheets()
	if len(sheets) == 0 {
		return nil, nil
	}
	for _, row := range sheets[0].GetRows() {
		var cells []string
		for _, cell := range row.GetCols() {
			cells = append(cells, cell.GetString())
		}
		rows = append(rows, cells)
	}
	return rows, nil
}

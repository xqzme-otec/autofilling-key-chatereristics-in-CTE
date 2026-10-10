package service

import (
	"context"
	"strings"
	"sync"

	"go.uber.org/zap"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/domain"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/logger"
)

// BatchItem is one product of an uploaded table and the outcome of its analysis: Session or Err.
type BatchItem struct {
	Row         int    // 1-based row of the table
	Name        string // empty when the table has no name for the product
	Description string
	Session     *domain.Session
	Err         error
}

// StartBatch starts a session for every product of a table, the same way Start does for a single
// description. A failed row does not fail the batch: its error is returned in the item.
func (s *Analysis) StartBatch(ctx context.Context, rows [][]string) ([]BatchItem, error) {
	items := products(rows)
	if len(items) == 0 {
		return nil, domain.Errorf(domain.CodeInvalidFile, "В файле не найдено товаров: у товара должно быть описание")
	}
	if len(items) > s.limits.MaxBatchItems {
		return nil, domain.Errorf(domain.CodeInvalidFile, "Товаров в файле: %d. За один раз обрабатывается не больше %d",
			len(items), s.limits.MaxBatchItems)
	}

	log := logger.FromContext(ctx)
	var wg sync.WaitGroup
	slots := make(chan struct{}, s.limits.BatchConcurrency)
	for i := range items {
		slots <- struct{}{}
		wg.Go(func() {
			defer func() { <-slots }()
			item := &items[i]
			item.Session, item.Err = s.Start(ctx, productText(item.Name, item.Description))
			if item.Err != nil {
				log.Warn("batch row failed", zap.Int("row", item.Row), zap.Error(item.Err))
			}
		})
	}
	wg.Wait()

	failed := 0
	for _, item := range items {
		if item.Err != nil {
			failed++
		}
	}
	log.Info("batch analyzed", zap.Int("rows", len(rows)), zap.Int("products", len(items)), zap.Int("failed", failed))
	return items, nil
}

// products picks the product rows of a table. A product is a row with a description; the name is
// optional, and a row with a name alone is not a product.
func products(rows [][]string) []BatchItem {
	nameCol, descCol, first := columns(rows)
	var items []BatchItem
	for i := first; i < len(rows); i++ {
		if description := cell(rows[i], descCol); description != "" {
			items = append(items, BatchItem{Row: i + 1, Name: cell(rows[i], nameCol), Description: description})
		}
	}
	return items
}

// Header prefixes, lower case. A description header is checked first: «Описание товара».
var (
	descriptionHeaders = []string{"описание", "характеристики", "description"}
	nameHeaders        = []string{"название", "наименование", "товар", "name"}
)

// columns finds the name and description columns (-1 = none) and the first data row. With a
// header row («Название», «Описание») the columns are taken by their titles. Without one they
// go by position: name, then description, or a single column of descriptions.
func columns(rows [][]string) (nameCol, descCol, first int) {
	for first < len(rows) && blankRow(rows[first]) {
		first++
	}
	if first == len(rows) {
		return -1, -1, first
	}

	nameCol, descCol = -1, -1
	for j := range rows[first] {
		title := strings.ToLower(cell(rows[first], j))
		switch {
		case descCol < 0 && hasAnyPrefix(title, descriptionHeaders):
			descCol = j
		case nameCol < 0 && hasAnyPrefix(title, nameHeaders):
			nameCol = j
		}
	}
	if descCol >= 0 {
		return nameCol, descCol, first + 1
	}

	// No header: the leftmost used column and, if any row has it, the one after.
	left, second := -1, false
	for _, row := range rows[first:] {
		for j := range row {
			if cell(row, j) != "" && (left < 0 || j < left) {
				left = j
			}
		}
	}
	for _, row := range rows[first:] {
		second = second || cell(row, left+1) != ""
	}
	if second {
		return left, left + 1, first
	}
	return -1, left, first
}

// cell is the trimmed text of column col, "" when the row has no such column.
func cell(row []string, col int) string {
	if col < 0 || col >= len(row) || blank(row[col]) {
		return ""
	}
	return strings.TrimSpace(row[col])
}

func blankRow(row []string) bool {
	for j := range row {
		if cell(row, j) != "" {
			return false
		}
	}
	return true
}

func hasAnyPrefix(s string, prefixes []string) bool {
	for _, p := range prefixes {
		if strings.HasPrefix(s, p) {
			return true
		}
	}
	return false
}

// productText is what goes to the model for a row: the name followed by the description, or the
// description alone when it already contains the name.
func productText(name, description string) string {
	if name == "" || strings.Contains(strings.ToLower(description), strings.ToLower(name)) {
		return description
	}
	return name + ", " + description
}

package service

import (
	"reflect"
	"testing"
)

func TestProducts(t *testing.T) {
	type product struct {
		row               int
		name, description string
	}
	cases := []struct {
		name string
		rows [][]string
		want []product
	}{
		{"header row, name without description is not a product", [][]string{
			{"Название", "Описание"},
			{"Стул", " на колесиках "},
			nil,
			{"", "молоток слесарный"},
			{"Халат детский", " "},
		}, []product{{2, "Стул", "на колесиках"}, {4, "", "молоток слесарный"}}},
		{"header picks the columns by title", [][]string{
			nil,
			{"№", "Описание товара", "Цена", "Наименование товара"},
			{"1", "на колесиках", "100", "Стул"},
		}, []product{{3, "Стул", "на колесиках"}}},
		{"header without a name column", [][]string{
			{"Артикул", "Characteristics", "Description"},
			{"A-1", "x", "стул офисный"},
		}, []product{{2, "", "стул офисный"}}},
		{"no header: name, then description", [][]string{
			{"Стул", "на колесиках"},
			{"Стол"},
			{"", "молоток слесарный"},
		}, []product{{1, "Стул", "на колесиках"}, {3, "", "молоток слесарный"}}},
		{"no header: a single column of descriptions", [][]string{
			{"стул офисный"},
			{""},
			{"молоток слесарный"},
		}, []product{{1, "", "стул офисный"}, {3, "", "молоток слесарный"}}},
		{"no header: the table starts in column B", [][]string{
			{"", "Стул", "на колесиках"},
			{"", "Стол", "письменный"},
		}, []product{{1, "Стул", "на колесиках"}, {2, "Стол", "письменный"}}},
		{"a description that mentions a header word is data", [][]string{
			{"стул с описанием на упаковке"},
		}, []product{{1, "", "стул с описанием на упаковке"}}},
		{"only a header", [][]string{{"Название", "Описание"}}, nil},
		{"empty table", nil, nil},
	}
	for _, c := range cases {
		var got []product
		for _, item := range products(c.rows) {
			got = append(got, product{item.Row, item.Name, item.Description})
		}
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s: got %+v, want %+v", c.name, got, c.want)
		}
	}
}

func TestProductText(t *testing.T) {
	cases := []struct{ name, description, want string }{
		{"Стул офисный", "на колесиках, кожзам", "Стул офисный, на колесиках, кожзам"},
		{"", "стул офисный на колесиках", "стул офисный на колесиках"},
		{"Стул офисный", "стул офисный на колесиках", "стул офисный на колесиках"},
	}
	for _, c := range cases {
		if got := productText(c.name, c.description); got != c.want {
			t.Errorf("productText(%q, %q) = %q, want %q", c.name, c.description, got, c.want)
		}
	}
}

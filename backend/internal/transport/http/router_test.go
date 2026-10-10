package http_test

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/internal/client/model"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/internal/repository"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/internal/service"
	httptransport "github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/internal/transport/http"
)

const (
	code1    = "31.01.11.150-00000003"
	code2    = "31.01.11.150-00000004"
	userText = "  стл на металлическом\n каркасе, ёмкость "
	cleanTxt = "стл на металлическом каркасе, емкость"
)

// modelResult mimics ml.pipeline.Pipeline.analyze(): one question, then the final code.
func modelResult(raw string, answers []any) map[string]any {
	done := false
	for _, a := range answers {
		done = done || a.(map[string]any)["characteristic"] == "Наличие подлокотников"
	}
	candidates := []any{map[string]any{"code": code1, "name": "Стул на металлическом каркасе", "score": 0.92}}
	var nextStep any
	if !done {
		candidates = append(candidates, map[string]any{"code": code2, "name": "Стул на металлическом каркасе", "score": 0.91})
		nextStep = map[string]any{"type": "characteristic", "message": "Уточните характеристику:",
			"characteristic": "Наличие подлокотников", "options": []string{"Да", "Нет"}, "required": true}
	}
	var ktru, ktruName any
	if done {
		ktru, ktruName = code1, "Стул на металлическом каркасе"
	}
	return map[string]any{
		"input": raw,
		"text":  strings.Replace(raw, "стл", "стул", 1),
		"corrections": []any{map[string]any{"from": "стл", "to": "стул", "span": []int{0, 3},
			"reason": "опечатка", "alternatives": []string{"стол"}}},
		"characteristics": []any{map[string]any{"name": "Тип каркаса", "value": "Металлический", "unit": nil,
			"source": "металлическом", "span": []int{7, 20}, "method": "stem", "confidence": 0.85}},
		"candidates": candidates, "candidates_total": len(candidates), "errors": []any{},
		"next_step": nextStep, "ktru": ktru, "ktru_name": ktruName,
		"steps":      []any{map[string]any{"stage": "Поиск по описанию", "count": 2}},
		"timings_ms": map[string]any{"correct": 0.3},
	}
}

// fakeModel stands in for the model service and records received payloads.
type fakeModel struct {
	mu       sync.Mutex
	requests []map[string]any
	// reply returns status and body (string = raw text, anything else = JSON).
	reply func(payload map[string]any) (int, any)
}

func (f *fakeModel) ok(p map[string]any) (int, any) {
	return 200, modelResult(p["raw"].(string), p["answers"].([]any))
}

func (f *fakeModel) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || r.URL.Path != "/analyze" {
		http.Error(w, "unexpected request", http.StatusTeapot)
		return
	}
	var payload map[string]any
	_ = json.NewDecoder(r.Body).Decode(&payload)
	f.mu.Lock()
	f.requests = append(f.requests, payload)
	f.mu.Unlock()
	status, body := f.reply(payload)
	w.WriteHeader(status)
	if s, ok := body.(string); ok {
		io.WriteString(w, s)
	} else {
		json.NewEncoder(w).Encode(body)
	}
}

type testEnv struct {
	api   *httptest.Server
	model *fakeModel
	redis *miniredis.Miniredis
}

// setup wires all real layers with a fake model service and in-memory Redis.
func setup(t *testing.T) *testEnv {
	fm := &fakeModel{}
	fm.reply = fm.ok
	modelSrv := httptest.NewServer(fm)
	t.Cleanup(modelSrv.Close)
	return setupWithModelURL(t, fm, modelSrv.URL)
}

func setupWithModelURL(t *testing.T, fm *fakeModel, modelURL string) *testEnv {
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr(), MaxRetries: -1})
	analysis := service.NewAnalysis(
		repository.NewRedisSessionRepository(rdb, time.Hour),
		model.NewClient(modelURL, 300*time.Millisecond),
		100,
	)
	api := httptest.NewServer(httptransport.NewRouter(analysis, []string{"*"}))
	t.Cleanup(api.Close)
	return &testEnv{api: api, model: fm, redis: mr}
}

func (e *testEnv) post(t *testing.T, path string, body any) (int, map[string]any) {
	t.Helper()
	raw, _ := json.Marshal(body)
	resp, err := http.Post(e.api.URL+path, "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, out
}

func (e *testEnv) start(t *testing.T) map[string]any {
	t.Helper()
	status, body := e.post(t, "/api/v1/analyze", map[string]string{"text": userText})
	if status != 200 {
		t.Fatalf("analyze: %d %v", status, body)
	}
	return body
}

func (e *testEnv) answer(t *testing.T, id, value string) (int, map[string]any) {
	return e.post(t, "/api/v1/sessions/"+id+"/answers", map[string]string{"value": value})
}

func jsonEq(t *testing.T, got any, want string) {
	t.Helper()
	var g, w any
	raw, _ := json.Marshal(got)
	json.Unmarshal(raw, &g)
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(g, w) {
		t.Errorf("got  %s\nwant %s", raw, want)
	}
}

func errCode(body map[string]any) any { return body["error"].(map[string]any)["code"] }

func TestHealth(t *testing.T) {
	resp, err := http.Get(setup(t).api.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	var body any
	json.NewDecoder(resp.Body).Decode(&body)
	jsonEq(t, body, `{"status": "ok"}`)
}

func TestAnalyzeSendsPreprocessedTextAndReturnsAgreedForm(t *testing.T) {
	e := setup(t)
	body := e.start(t)

	id := body["session_id"].(string)
	delete(body, "session_id")
	if id == "" {
		t.Fatal("empty session_id")
	}
	jsonEq(t, body, `{
		"corrections": [{"from": "стл", "to": "стул", "reason": "опечатка"}],
		"characteristics": [{"name": "Тип каркаса", "value": "Металлический", "source": "металлическом"}],
		"candidates": [
			{"code": "`+code1+`", "name": "Стул на металлическом каркасе", "score": 0.92},
			{"code": "`+code2+`", "name": "Стул на металлическом каркасе", "score": 0.91}],
		"errors": [],
		"next_step": {"message": "Уточните характеристику:", "characteristic": "Наличие подлокотников", "options": ["Да", "Нет"]},
		"ktru": null, "ktru_name": null,
		"steps": [{"stage": "Поиск по описанию", "count": 2}]}`)
	jsonEq(t, e.model.requests, `[{"raw": "`+cleanTxt+`", "answers": []}]`)
}

func TestCharacteristicUnitAndErrorFields(t *testing.T) {
	e := setup(t)
	e.model.reply = func(p map[string]any) (int, any) {
		_, res := e.model.ok(p)
		r := res.(map[string]any)
		r["characteristics"] = []any{map[string]any{"name": "Высота", "value": "1000–1200", "unit": "Миллиметр",
			"user_value": "1100 Миллиметр", "source": "110 см", "span": []int{0, 6}}}
		r["errors"] = []any{map[string]any{"type": "nonexistent_value", "characteristic": "Материал",
			"value": "дерево", "message": "Значение недопустимо", "suggestion": "Металл"}}
		r["corrections"] = nil
		return 200, r
	}
	body := e.start(t)

	jsonEq(t, body["characteristics"], `[{"name": "Высота", "value": "1000–1200", "unit": "Миллиметр", "source": "110 см"}]`)
	jsonEq(t, body["errors"], `[{"type": "nonexistent_value", "message": "Значение недопустимо",
		"characteristic": "Материал", "suggestion": "Металл"}]`)
	jsonEq(t, body["corrections"], `[]`)
}

func TestAnswerResendsTextWithAllAnswers(t *testing.T) {
	e := setup(t)
	id := e.start(t)["session_id"].(string)

	status, body := e.answer(t, id, "Нет")

	if status != 200 {
		t.Fatalf("%d %v", status, body)
	}
	if body["session_id"] != id || body["next_step"] != nil || body["ktru"] != code1 {
		t.Errorf("unexpected body %v", body)
	}
	jsonEq(t, e.model.requests[1], `{"raw": "`+cleanTxt+`",
		"answers": [{"type": "characteristic", "characteristic": "Наличие подлокотников", "value": "Нет"}]}`)
}

func TestProductAnswerHasNoCharacteristic(t *testing.T) {
	e := setup(t)
	e.model.reply = func(p map[string]any) (int, any) {
		_, res := e.model.ok(p)
		res.(map[string]any)["next_step"] = map[string]any{"type": "product", "message": "Какой товар?",
			"characteristic": nil, "options": []string{"Стул", "Табурет"}}
		return 200, res
	}
	id := e.start(t)["session_id"].(string)

	if status, body := e.answer(t, id, "Табурет"); status != 200 {
		t.Fatalf("%d %v", status, body)
	}
	jsonEq(t, e.model.requests[1]["answers"], `[{"type": "product", "value": "Табурет"}]`)
}

func TestAnswerErrors(t *testing.T) {
	e := setup(t)
	id := e.start(t)["session_id"].(string)

	cases := []struct {
		name, id string
		body     any
		status   int
		code     string
	}{
		{"not an option", id, map[string]string{"value": "Может быть"}, 422, "invalid_answer"},
		{"missing value", id, map[string]string{}, 422, "validation_error"},
		{"empty value", id, map[string]string{"value": ""}, 422, "validation_error"},
		{"unknown session", "nope", map[string]string{"value": "Нет"}, 404, "session_not_found"},
	}
	for _, c := range cases {
		status, body := e.post(t, "/api/v1/sessions/"+c.id+"/answers", c.body)
		if status != c.status || errCode(body) != c.code {
			t.Errorf("%s: got %d %v", c.name, status, body)
		}
	}

	e.answer(t, id, "Нет")
	if status, body := e.answer(t, id, "Нет"); status != 409 || errCode(body) != "no_pending_question" {
		t.Errorf("answer after done: %d %v", status, body)
	}
}

func TestInvalidText(t *testing.T) {
	e := setup(t)
	cases := []struct {
		body any
		code string
	}{
		{map[string]any{}, "validation_error"},
		{map[string]any{"text": ""}, "validation_error"},
		{map[string]any{"text": 5}, "validation_error"},
		{"not an object", "validation_error"},
		{map[string]any{"text": " \n\t\xe2\x80\x8b "}, "invalid_text"}, // whitespace + zero-width space
		{map[string]any{"text": strings.Repeat("я", 101)}, "invalid_text"},
	}
	for _, c := range cases {
		if status, body := e.post(t, "/api/v1/analyze", c.body); status != 422 || errCode(body) != c.code {
			t.Errorf("%v: got %d %v", c.body, status, body)
		}
	}
	resp, err := http.Post(e.api.URL+"/api/v1/analyze", "application/json", strings.NewReader(`{"text": "стул"} trailing`))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 422 {
		t.Errorf("data after the JSON object: got %d, want 422", resp.StatusCode)
	}
	if len(e.model.requests) != 0 {
		t.Errorf("model was called: %v", e.model.requests)
	}
}

func TestModelFailures(t *testing.T) {
	cases := []struct {
		name   string
		reply  func(map[string]any) (int, any)
		status int
		code   string
	}{
		{"timeout", func(map[string]any) (int, any) { time.Sleep(time.Second); return 200, "{}" }, 504, "model_timeout"},
		{"http 500", func(map[string]any) (int, any) { return 500, "boom" }, 502, "model_bad_response"},
		{"not json", func(map[string]any) (int, any) { return 200, "not json" }, 502, "model_bad_response"},
		{"unexpected", func(map[string]any) (int, any) { return 200, map[string]any{"unexpected": true} }, 502, "model_bad_response"},
		{"bad next_step", func(map[string]any) (int, any) {
			return 200, map[string]any{"input": "x", "text": "x", "next_step": map[string]any{"type": "other"}}
		}, 502, "model_bad_response"},
	}
	for _, c := range cases {
		e := setup(t)
		e.model.reply = c.reply
		if status, body := e.post(t, "/api/v1/analyze", map[string]string{"text": userText}); status != c.status || errCode(body) != c.code {
			t.Errorf("%s: got %d %v", c.name, status, body)
		}
	}

	e := setupWithModelURL(t, &fakeModel{}, "http://127.0.0.1:1")
	if status, body := e.post(t, "/api/v1/analyze", map[string]string{"text": userText}); status != 503 || errCode(body) != "model_unavailable" {
		t.Errorf("unavailable: got %d %v", status, body)
	}
}

func TestStorageUnavailable(t *testing.T) {
	e := setup(t)
	e.redis.Close()
	if status, body := e.post(t, "/api/v1/analyze", map[string]string{"text": userText}); status != 503 || errCode(body) != "storage_unavailable" {
		t.Errorf("got %d %v", status, body)
	}
}

func TestCORS(t *testing.T) {
	h := httptransport.NewRouter(nil, []string{"http://allowed"})

	req := httptest.NewRequest(http.MethodOptions, "/api/v1/analyze", nil)
	req.Header.Set("Origin", "http://allowed")
	req.Header.Set("Access-Control-Request-Method", "POST")
	req.Header.Set("Access-Control-Request-Headers", "content-type")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 204 || rec.Header().Get("Access-Control-Allow-Origin") != "http://allowed" ||
		rec.Header().Get("Access-Control-Allow-Headers") != "content-type" {
		t.Errorf("preflight: got %d %v", rec.Code, rec.Header())
	}

	req = httptest.NewRequest(http.MethodGet, "/health", nil)
	req.Header.Set("Origin", "http://other")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 200 || rec.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Errorf("foreign origin: got %d %v", rec.Code, rec.Header())
	}
}

func TestUnknownRouteUsesErrorFormat(t *testing.T) {
	resp, err := http.Get(setup(t).api.URL + "/api/v1/nope")
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	json.NewDecoder(resp.Body).Decode(&body)
	if resp.StatusCode != 404 || errCode(body) != "not_found" {
		t.Errorf("got %d %v", resp.StatusCode, body)
	}
}

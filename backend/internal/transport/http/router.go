// Package http is the HTTP API of the aggregator for the frontend.
package http

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"slices"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/internal/service"
)

const maxBodyBytes = 1 << 20

type handler struct {
	analysis *service.Analysis
}

func NewRouter(analysis *service.Analysis, corsOrigins []string) http.Handler {
	h := &handler{analysis: analysis}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/v1/analyze", h.analyze)
	mux.HandleFunc("POST /api/v1/sessions/{id}/answers", h.answer)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		writeProblem(w, http.StatusNotFound, "not_found", "Not Found", nil)
	})
	return cors(corsOrigins, mux)
}

// analyze starts a new session from a free-form product description.
func (h *handler) analyze(w http.ResponseWriter, r *http.Request) {
	text, ok := readField(w, r, "text")
	if !ok {
		return
	}
	session, err := h.analysis.Start(r.Context(), text)
	respond(w, session, err)
}

// answer answers the clarifying question (next_step) of the session with one of its options.
func (h *handler) answer(w http.ResponseWriter, r *http.Request) {
	value, ok := readField(w, r, "value")
	if !ok {
		return
	}
	session, err := h.analysis.Answer(r.Context(), r.PathValue("id"), value)
	respond(w, session, err)
}

// readField reads a required non-empty string field from a JSON object body;
// on failure it writes the validation error and returns false.
func readField(w http.ResponseWriter, r *http.Request, name string) (string, bool) {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	var body map[string]any
	err := dec.Decode(&body)
	if err == nil && dec.Decode(&struct{}{}) != io.EOF {
		err = errors.New("unexpected data after the JSON object")
	}
	if err != nil {
		writeProblem(w, http.StatusUnprocessableEntity, "validation_error", "Request validation failed",
			[]string{"body: invalid JSON object: " + err.Error()})
		return "", false
	}
	value, ok := body[name].(string)
	if !ok || value == "" {
		writeProblem(w, http.StatusUnprocessableEntity, "validation_error", "Request validation failed",
			[]string{name + ": required non-empty string"})
		return "", false
	}
	return value, true
}

func cors(origins []string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin != "" && (slices.Contains(origins, "*") || slices.Contains(origins, origin)) {
			h := w.Header()
			h.Set("Access-Control-Allow-Origin", origin)
			h.Add("Vary", "Origin")
			if r.Method == http.MethodOptions && r.Header.Get("Access-Control-Request-Method") != "" {
				h.Set("Access-Control-Allow-Methods", "GET, POST")
				h.Set("Access-Control-Allow-Headers", r.Header.Get("Access-Control-Request-Headers"))
				w.WriteHeader(http.StatusNoContent)
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

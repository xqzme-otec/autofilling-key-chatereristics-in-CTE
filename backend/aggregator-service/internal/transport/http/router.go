// Package http is the HTTP API of the aggregator for the frontend.
package http

import (
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"slices"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/logger"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/service"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/sheet"
)

type handler struct {
	analysis     *service.Analysis
	maxBodyBytes int64
}

func NewRouter(analysis *service.Analysis, corsOrigins []string, maxBodyBytes int64) http.Handler {
	h := &handler{analysis: analysis, maxBodyBytes: maxBodyBytes}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/v1/analyze", h.analyze)
	mux.HandleFunc("POST /api/v1/analyze/batch", h.analyzeBatch)
	mux.HandleFunc("POST /api/v1/sessions/{id}/answers", h.answer)
	mux.HandleFunc("GET /docs", docs)
	mux.HandleFunc("GET "+openAPIPath, openAPI)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		writeProblem(w, http.StatusNotFound, "not_found", "Маршрут не найден", nil)
	})
	return withRequestLog(cors(corsOrigins, mux))
}

// analyze starts a new session from a free-form product description.
func (h *handler) analyze(w http.ResponseWriter, r *http.Request) {
	text, ok := h.readField(w, r, "text")
	if !ok {
		return
	}
	session, err := h.analysis.Start(r.Context(), text)
	respond(w, session, err)
}

// analyzeBatch starts a session for every product of an uploaded table: a multipart/form-data
// body with the .csv, .xlsx or .xls file in the "file" field.
func (h *handler) analyzeBatch(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, h.maxBodyBytes)
	// The whole body is under the limit, so the file stays in memory: no temporary files.
	if err := r.ParseMultipartForm(h.maxBodyBytes); err != nil {
		if tooLarge := new(http.MaxBytesError); errors.As(err, &tooLarge) {
			logError(w, err)
			writeProblem(w, http.StatusRequestEntityTooLarge, "file_too_large",
				fmt.Sprintf("Файл слишком большой: допускается не больше %d КБ", h.maxBodyBytes>>10), nil)
			return
		}
		invalidRequest(w, "body: invalid multipart/form-data: "+err.Error())
		return
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		invalidRequest(w, "file: required file field")
		return
	}
	defer file.Close()
	data, err := io.ReadAll(file)
	if err != nil {
		writeError(w, err)
		return
	}

	rows, err := sheet.Read(header.Filename, data)
	if err != nil {
		writeError(w, err)
		return
	}
	items, err := h.analysis.StartBatch(r.Context(), rows)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, newBatchResponse(items))
}

// answer answers the clarifying question (next_step) of the session with one of its options.
func (h *handler) answer(w http.ResponseWriter, r *http.Request) {
	value, ok := h.readField(w, r, "value")
	if !ok {
		return
	}
	session, err := h.analysis.Answer(r.Context(), r.PathValue("id"), value)
	respond(w, session, err)
}

// readField reads a required non-empty string field from a JSON object body;
// on failure it writes the validation error and returns false.
func (h *handler) readField(w http.ResponseWriter, r *http.Request, name string) (string, bool) {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, h.maxBodyBytes))
	var body map[string]any
	err := dec.Decode(&body)
	if err == nil && dec.Decode(&struct{}{}) != io.EOF {
		err = errors.New("unexpected data after the JSON object")
	}
	if err != nil {
		invalidRequest(w, "body: invalid JSON object: "+err.Error())
		return "", false
	}
	value, ok := body[name].(string)
	if !ok || value == "" {
		invalidRequest(w, name+": required non-empty string")
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

// validRequestID is the rule of the model service (ml/service/middleware.py): a client-provided
// ID goes to the logs as is, so only safe characters are accepted.
var validRequestID = regexp.MustCompile(`^[A-Za-z0-9._:-]{1,128}$`)

// withRequestLog gives every request an ID (a valid X-Request-ID from the client or a new one),
// returns it in the X-Request-ID header and writes one log line per request, an error for 5xx.
func withRequestLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		id := r.Header.Get(logger.RequestIDHeader)
		if !validRequestID.MatchString(id) {
			id = rand.Text()
		}
		w.Header().Set(logger.RequestIDHeader, id)
		ctx := logger.WithRequestID(r.Context(), id)
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r.WithContext(ctx))

		level := zapcore.InfoLevel
		if rec.status >= http.StatusInternalServerError {
			level = zapcore.ErrorLevel
		}
		logger.FromContext(ctx).Log(level, "request", zap.String("method", r.Method), zap.String("path", r.URL.Path),
			zap.Int("status", rec.status), zap.Float64("duration_ms", float64(time.Since(start).Microseconds())/1000),
			zap.Error(rec.err))
	})
}

// statusRecorder keeps what the request log line reports: the response status and, via
// logError, why the request failed.
type statusRecorder struct {
	http.ResponseWriter
	status int
	err    error
}

func (r *statusRecorder) WriteHeader(status int) {
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

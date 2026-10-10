package http

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/domain"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/logger"
)

// sessionResponse is a reduced view of the model output agreed with the frontend: only the
// fields the UI shows, plus session_id to continue the dialog.
type sessionResponse struct {
	SessionID       string                  `json:"session_id"`
	Corrections     []domain.Correction     `json:"corrections"`
	Characteristics []domain.Characteristic `json:"characteristics"`
	ProductIntent   *string                 `json:"product_intent,omitempty"`
	SubtypeContext  []domain.ContextSignal  `json:"subtype_context,omitempty"`
	RetrievalCount  *int                    `json:"retrieval_count,omitempty"`
	CompatibleCount *int                    `json:"compatible_count,omitempty"`
	RemainingCount  *int                    `json:"remaining_count,omitempty"`
	Candidates      []domain.Candidate      `json:"candidates"`
	CandidatesTotal int                     `json:"candidates_total"`
	Errors          []domain.AnalysisError  `json:"errors"`
	MissingRequired []string                `json:"missing_required"`
	NextStep        *nextStepResponse       `json:"next_step"`
	NextQuestion    *nextStepResponse       `json:"next_question,omitempty"`
	Ktru            *string                 `json:"ktru"`
	KtruName        *string                 `json:"ktru_name"`
	Steps           []domain.Step           `json:"steps"`
}

type nextStepResponse struct {
	Message        string   `json:"message"`
	Characteristic *string  `json:"characteristic"`
	Options        []string `json:"options"`
}

func newSessionResponse(s *domain.Session) sessionResponse {
	r := s.Result
	out := sessionResponse{
		SessionID:       s.ID,
		Corrections:     nonNil(r.Corrections),
		Characteristics: nonNil(r.Characteristics),
		ProductIntent:   r.ProductIntent,
		SubtypeContext:  r.SubtypeContext,
		RetrievalCount:  r.RetrievalCount,
		CompatibleCount: r.CompatibleCount,
		RemainingCount:  r.RemainingCount,
		Candidates:      nonNil(r.Candidates),
		CandidatesTotal: r.CandidatesTotal,
		Errors:          nonNil(r.Errors),
		MissingRequired: nonNil(r.MissingRequired),
		Ktru:            r.Ktru,
		KtruName:        r.KtruName,
		Steps:           nonNil(r.Steps),
	}
	if q := r.NextStep; q != nil {
		out.NextStep = &nextStepResponse{Message: *q.Message, Characteristic: q.Characteristic, Options: q.Options}
	}
	if q := r.NextQuestion; q != nil {
		out.NextQuestion = &nextStepResponse{Message: *q.Message, Characteristic: q.Characteristic, Options: q.Options}
	}
	return out
}

func nonNil[T any](s []T) []T {
	if s == nil {
		return []T{}
	}
	return s
}

var statusByCode = map[domain.ErrorCode]int{
	domain.CodeInvalidText:        http.StatusUnprocessableEntity,
	domain.CodeInvalidAnswer:      http.StatusUnprocessableEntity,
	domain.CodeSessionNotFound:    http.StatusNotFound,
	domain.CodeNoPendingQuestion:  http.StatusConflict,
	domain.CodeModelBadResponse:   http.StatusBadGateway,
	domain.CodeModelUnavailable:   http.StatusServiceUnavailable,
	domain.CodeStorageUnavailable: http.StatusServiceUnavailable,
	domain.CodeModelTimeout:       http.StatusGatewayTimeout,
}

func respond(w http.ResponseWriter, session *domain.Session, err error) {
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, newSessionResponse(session))
}

// writeError maps a domain error to its HTTP status; anything else is a 500. The client gets the
// code and the short message only; the full error with its cause goes to the request log line.
func writeError(w http.ResponseWriter, err error) {
	logError(w, err)
	var de *domain.Error
	if !errors.As(err, &de) {
		writeProblem(w, http.StatusInternalServerError, "internal_error", "Внутренняя ошибка сервиса", nil)
		return
	}
	status, ok := statusByCode[de.Code]
	if !ok {
		status = http.StatusInternalServerError
	}
	writeProblem(w, status, string(de.Code), de.Message, nil)
}

// invalidRequest writes a validation_error; detail says what is wrong with the request body.
func invalidRequest(w http.ResponseWriter, detail string) {
	logError(w, errors.New(detail))
	writeProblem(w, http.StatusUnprocessableEntity, "validation_error", "Некорректный запрос", []string{detail})
}

// logError attaches err to the log line of the request (see withRequestLog).
func logError(w http.ResponseWriter, err error) {
	if rec, ok := w.(*statusRecorder); ok {
		rec.err = errors.Join(rec.err, err)
	}
}

// writeProblem writes the single error body format: {"error": {"code", "message", "request_id", "details"?}}.
// request_id is the X-Request-ID that withRequestLog has set on the response.
func writeProblem(w http.ResponseWriter, status int, code, message string, details []string) {
	body := map[string]any{"code": code, "message": message, "request_id": w.Header().Get(logger.RequestIDHeader)}
	if details != nil {
		body["details"] = details
	}
	writeJSON(w, status, map[string]any{"error": body})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		logError(w, fmt.Errorf("write response: %w", err))
	}
}

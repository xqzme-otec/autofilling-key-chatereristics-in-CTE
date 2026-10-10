package http

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/internal/domain"
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
	Errors          []domain.AnalysisError  `json:"errors"`
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
		Errors:          nonNil(r.Errors),
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

// writeError maps a domain error to its HTTP status; anything else is a 500.
func writeError(w http.ResponseWriter, err error) {
	var de *domain.Error
	if !errors.As(err, &de) {
		slog.Error("internal error", "err", err)
		writeProblem(w, http.StatusInternalServerError, "internal_error", "Internal server error", nil)
		return
	}
	status, ok := statusByCode[de.Code]
	if !ok {
		status = http.StatusInternalServerError
	}
	if status >= 500 {
		slog.Error(string(de.Code), "message", de.Message)
	}
	writeProblem(w, status, string(de.Code), de.Message, nil)
}

// writeProblem writes the single error body format: {"error": {"code", "message", "details"?}}.
func writeProblem(w http.ResponseWriter, status int, code, message string, details []string) {
	body := map[string]any{"code": code, "message": message}
	if details != nil {
		body["details"] = details
	}
	writeJSON(w, status, map[string]any{"error": body})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		slog.Error("write response", "err", err)
	}
}

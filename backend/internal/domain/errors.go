package domain

import "fmt"

// ErrorCode is the machine-readable error code returned to clients.
type ErrorCode string

const (
	CodeInvalidText        ErrorCode = "invalid_text"
	CodeInvalidAnswer      ErrorCode = "invalid_answer"
	CodeSessionNotFound    ErrorCode = "session_not_found"
	CodeNoPendingQuestion  ErrorCode = "no_pending_question"
	CodeModelUnavailable   ErrorCode = "model_unavailable"
	CodeModelTimeout       ErrorCode = "model_timeout"
	CodeModelBadResponse   ErrorCode = "model_bad_response"
	CodeStorageUnavailable ErrorCode = "storage_unavailable"
)

type Error struct {
	Code    ErrorCode
	Message string
}

func (e *Error) Error() string { return string(e.Code) + ": " + e.Message }

func Errorf(code ErrorCode, format string, args ...any) *Error {
	return &Error{Code: code, Message: fmt.Sprintf(format, args...)}
}

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

// Error is an expected failure. Message is shown to the user as is: short, in Russian, without
// internal details. Cause, if any, goes only to the logs.
type Error struct {
	Code    ErrorCode
	Message string
	Cause   error
}

func (e *Error) Error() string {
	if e.Cause == nil {
		return string(e.Code) + ": " + e.Message
	}
	return string(e.Code) + ": " + e.Message + ": " + e.Cause.Error()
}

func (e *Error) Unwrap() error { return e.Cause }

func Errorf(code ErrorCode, format string, args ...any) *Error {
	return &Error{Code: code, Message: fmt.Sprintf(format, args...)}
}

// Wrap returns an Error whose cause is logged but never shown to the user.
func Wrap(code ErrorCode, cause error, message string) *Error {
	return &Error{Code: code, Message: message, Cause: cause}
}

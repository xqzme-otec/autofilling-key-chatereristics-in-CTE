// Package repository is the data access layer: the service depends on the interfaces declared
// here, implementations hide the storage behind them.
package repository

import (
	"context"
	"errors"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/domain"
)

// ErrSessionNotFound is returned when a session does not exist or has expired.
var ErrSessionNotFound = errors.New("session not found")

type SessionRepository interface {
	Get(ctx context.Context, id string) (*domain.Session, error)
	// Save creates or replaces the session.
	Save(ctx context.Context, session *domain.Session) error
}

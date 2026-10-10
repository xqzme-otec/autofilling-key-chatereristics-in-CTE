// Package service implements the analysis dialog: validate the text -> ask the model -> keep the
// session for follow-up answers. The text goes to the model as typed: normalization, typos and the
// keyboard layout are fixed by the model's pipeline.
//
// The model service is stateless: every call receives the full text and all answers given so far.
// The session keeps them so the frontend only sends the next answer.
package service

import (
	"context"
	"crypto/rand"
	"errors"
	"slices"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"go.uber.org/zap"

	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/domain"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/logger"
	"github.com/xqzme-otec/autofilling-key-chatereristics-in-CTE/backend/aggregator-service/internal/repository"
)

type Model interface {
	Analyze(ctx context.Context, text string, answers []domain.Answer) (*domain.AnalysisResult, error)
}

type Limits struct {
	MaxTextLength    int // characters in a product description
	MaxBatchItems    int // products in an uploaded table
	BatchConcurrency int // products of a table analyzed at the same time
}

type Analysis struct {
	sessions repository.SessionRepository
	model    Model
	limits   Limits
}

func NewAnalysis(sessions repository.SessionRepository, model Model, limits Limits) *Analysis {
	return &Analysis{sessions: sessions, model: model, limits: limits}
}

// Start creates a session from a free-form product description.
func (s *Analysis) Start(ctx context.Context, text string) (*domain.Session, error) {
	if utf8.RuneCountInString(text) > s.limits.MaxTextLength {
		return nil, domain.Errorf(domain.CodeInvalidText, "Описание длиннее %d символов", s.limits.MaxTextLength)
	}
	if blank(text) {
		return nil, domain.Errorf(domain.CodeInvalidText, "Описание товара пустое")
	}

	result, err := s.model.Analyze(ctx, text, []domain.Answer{})
	if err != nil {
		return nil, err
	}
	now := time.Now().UTC()
	session := &domain.Session{ID: rand.Text(), Text: text, Result: *result, Answers: []domain.Answer{},
		CreatedAt: now, UpdatedAt: now}
	if err := s.sessions.Save(ctx, session); err != nil {
		return nil, err
	}
	logger.FromContext(ctx).Info("session started", zap.String("session_id", session.ID),
		zap.String("status", string(session.Status())), zap.Int("candidates", result.CandidatesTotal))
	return session, nil
}

// blank: nothing but spaces and invisible characters (zero-width space, soft hyphen, control characters).
func blank(s string) bool {
	return strings.IndexFunc(s, func(r rune) bool {
		return !unicode.IsSpace(r) && !unicode.IsControl(r) && !unicode.Is(unicode.Cf, r)
	}) < 0
}

// Answer answers the pending question of the session with one of its options.
func (s *Analysis) Answer(ctx context.Context, id, value string) (*domain.Session, error) {
	session, err := s.sessions.Get(ctx, id)
	if errors.Is(err, repository.ErrSessionNotFound) {
		return nil, domain.Errorf(domain.CodeSessionNotFound, "Сессия не найдена или устарела. Начните подбор заново")
	}
	if err != nil {
		return nil, err
	}
	answer, err := toAnswer(session.Result.NextStep, value)
	if err != nil {
		return nil, err
	}

	answers := append(slices.Clone(session.Answers), answer)
	result, err := s.model.Analyze(ctx, session.Text, answers)
	if err != nil {
		return nil, err
	}
	session.Answers, session.Result, session.UpdatedAt = answers, *result, time.Now().UTC()
	if err := s.sessions.Save(ctx, session); err != nil {
		return nil, err
	}
	logger.FromContext(ctx).Info("session answered", zap.String("session_id", session.ID),
		zap.String("answer_type", answer.Type), zap.String("status", string(session.Status())),
		zap.Int("candidates", result.CandidatesTotal))
	return session, nil
}

// toAnswer: the frontend sends only the chosen option; the question type comes from the last next_step.
func toAnswer(q *domain.NextStep, value string) (domain.Answer, error) {
	if q == nil {
		return domain.Answer{}, domain.Errorf(domain.CodeNoPendingQuestion, "В сессии нет вопроса, ожидающего ответа")
	}
	if !slices.Contains(q.Options, value) {
		return domain.Answer{}, domain.Errorf(domain.CodeInvalidAnswer, "Выберите один из предложенных вариантов")
	}
	a := domain.Answer{Type: q.Type, Value: value}
	if q.Type == domain.AnswerCharacteristic && q.Characteristic != nil {
		a.Characteristic = *q.Characteristic
	}
	return a, nil
}
